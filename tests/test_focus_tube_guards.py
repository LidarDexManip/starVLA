"""FOCUS tube guards (2026-11-12): acquire only where the tube can be, drop a track that walks off, and
re-prompt SAM when the detector keeps finding the tube elsewhere. Run with the serving Python (unittest, CPU
only, no weights: the detector and SAM are faked)."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deployment.model_server import focus_view as fv

TUBE = [410.0, 440.0, 480.0, 505.0]          # the opening in the left hand (centre 445, 472)
PLUNGER = [725.0, 135.0, 825.0, 250.0]       # the pipette's blue plunger, right hand raised (#23 ep 1 / 6)
BLOCK = [845.0, 443.0, 911.0, 501.0]         # the orange-labelled block on the right (#23 ep 60)


def tracker(dets, sam_boxes):
    """dets: frame -> [(box, score)] the tube detector returns; sam_boxes: the boxes SAM tracks, in order."""
    t = fv.FocusTracker.__new__(fv.FocusTracker)
    t._owl_text = lambda rgb, text: dets(rgb)
    t.sam_calls = []

    def sam(rgb, box=None):
        t.sam_calls.append(box)
        b = sam_boxes.pop(0) if sam_boxes else t.sam_calls[-1]
        return [int(v) for v in b[:4]] + [int((b[2] - b[0]) * (b[3] - b[1]))]
    t._sam = sam
    t.reset("tube")
    return t


RGB = np.zeros((720, 1280, 3), np.uint8)


class TubeGuardTest(unittest.TestCase):
    def test_acquisition_skips_a_higher_scoring_box_off_the_tube_region(self):
        for wrong in (PLUNGER, BLOCK):
            t = tracker(lambda rgb: [(wrong, 0.4), (TUBE, 0.2)], [])
            self.assertEqual(t.detect(RGB), TUBE)
        t = tracker(lambda rgb: [(PLUNGER, 0.4), (BLOCK, 0.3)], [])
        self.assertIsNone(t.detect(RGB))                       # nothing plausible: wait for the next frame
        rec = t.step(RGB)
        self.assertFalse(rec["locked"])
        cx, cy = fv.PRIOR_CENTRE["tube"]
        self.assertEqual(rec["window"][:2], [cx - fv.SIDE / 2, cy - fv.SIDE / 2])

    def test_a_track_that_leaves_the_region_is_dropped_and_re_detected(self):
        t = tracker(lambda rgb: [(TUBE, 0.3)], [TUBE, TUBE, BLOCK, TUBE])
        self.assertTrue(t.step(RGB, t=0.0)["acquired"])
        self.assertTrue(t.step(RGB, t=0.1)["following"])
        rec = t.step(RGB, t=0.2)                               # SAM slid onto the block
        self.assertTrue(rec["dropped"]) ; self.assertFalse(rec["locked"]) ; self.assertIsNone(rec["box"])
        rec = t.step(RGB, t=0.3)
        self.assertTrue(rec["acquired"]) ; self.assertEqual(t.sam_calls[-1], TUBE)

    def test_recheck_re_prompts_only_after_two_disagreements_with_a_box_elsewhere(self):
        dets = {"d": [(TUBE, 0.3)]}
        t = tracker(lambda rgb: dets["d"], [[600, 400, 660, 460]] * 10)    # SAM locked on something else in region
        t.step(RGB, t=0.0)
        self.assertIsNone(t.recheck(RGB))                      # 1st disagreement
        t.step(RGB, t=0.1)
        box = t.recheck(RGB)                                   # 2nd: re-prompt at the detection
        self.assertEqual(box, TUBE)
        rec = t.step(RGB, t=0.2, reprompt=box)
        self.assertTrue(rec["reacquired"]) ; self.assertEqual(t.sam_calls[-1], TUBE)
        self.assertEqual(len(t.areas), 1)                      # the visibility reference restarts with the new prompt

    def test_recheck_keeps_the_track_when_the_detector_agrees_or_sees_nothing(self):
        dets = {"d": [(TUBE, 0.3)]}
        t = tracker(lambda rgb: dets["d"], [TUBE] * 20)
        t.step(RGB, t=0.0)
        for i in range(4):
            self.assertIsNone(t.recheck(RGB)) ; t.step(RGB, t=0.1 * (i + 1))
        dets["d"] = []                                         # pipette tip covering the opening
        for i in range(4):
            self.assertIsNone(t.recheck(RGB))
        # one disagreement, then agreement again: the count restarts
        t.box_now = [600, 400, 660, 460, 3600]
        dets["d"] = [(TUBE, 0.3)]
        self.assertIsNone(t.recheck(RGB))
        t.box_now = TUBE + [4550]
        self.assertIsNone(t.recheck(RGB))
        t.box_now = [600, 400, 660, 460, 3600]
        self.assertIsNone(t.recheck(RGB))

    def test_recheck_is_tube_only(self):
        t = tracker(lambda rgb: [(TUBE, 0.3)], [])
        t.reset("rack")
        t.locked = True
        self.assertIsNone(t.recheck(RGB))


if __name__ == "__main__":
    unittest.main()
