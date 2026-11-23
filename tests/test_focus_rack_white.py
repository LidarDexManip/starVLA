"""FOCUS rack detector (2026-11-16, new desk): the rack is white with blue holes. The text "a white test tube rack"
alone ranks the yellow tip box first, and the old largest-blue-blob rule locked onto the blue centrifuge lid, so a
rack box must also hold the rack's blue holes -- at least three round ones; 11-17 the text became "a white board
with blue holes", which ranks the rack first (11-17: a box over the tip box reached
the old 3 % blue share from the right gripper's blue parts and SAM slid onto the centrifuge). Run with the serving Python (unittest, CPU only, the detector faked)."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deployment.model_server import focus_view as fv

TIPBOX = [673.0, 478.0, 818.0, 659.0]   # yellow tip box, the detector's top "white test tube rack" box (live frame 11-16)
RACK = [396.0, 384.0, 644.0, 718.0]     # the white rack, second
LID = (625, 293, 768, 372)              # the centrifuge's blue lid, the largest blue blob in that frame
BLUE = (0, 0, 255)                      # RGB pure blue: inside the HSV blue band


def scene(holes=True):
    rgb = np.full((720, 1280, 3), 200, np.uint8)
    x0, y0, x1, y1 = LID
    rgb[y0:y1, x0:x1] = BLUE
    if holes:
        for cx in (470, 540):
            for cy in range(430, 700, 60):
                rgb[cy - 18:cy + 18, cx - 18:cx + 18] = BLUE
    return rgb


def tracker(dets):
    t = fv.FocusTracker.__new__(fv.FocusTracker)
    t.prompts = []
    t._owl_text = lambda rgb, text: (t.prompts.append(text), dets)[1]
    t.target = "rack"
    return t


class RackDetector(unittest.TestCase):
    def test_takes_the_box_with_blue_holes_over_a_higher_scored_tip_box(self):
        t = tracker([(TIPBOX, 0.145), (RACK, 0.061)])
        self.assertEqual(t.detect(scene()), RACK)
        self.assertEqual(t.prompts, ["a white board with blue holes"])

    def test_never_falls_back_to_the_largest_blue_blob(self):
        self.assertIsNone(tracker([(TIPBOX, 0.145)]).detect(scene()))
        self.assertIsNone(tracker([]).detect(scene()))

    def test_blue_that_is_not_hole_shaped_does_not_make_a_rack(self):
        # 11-17 episode 51: the tip-box box held ~3 % blue, all from the right gripper's long / ragged blue parts
        rgb = scene(holes=False)
        rgb[480:500, 690:810] = BLUE                     # a long bar (aspect 6)
        rgb[540:640, 700:800] = BLUE                     # a large blob (10000 px)
        rgb[600:660, 680:740:4] = BLUE                   # a ragged comb (fill well under 0.5)
        self.assertIsNone(tracker([(TIPBOX, 0.145)]).detect(rgb))

    def test_two_visible_holes_are_not_enough(self):
        rgb = scene(holes=False)
        for cy in (430, 490):
            rgb[cy - 18:cy + 18, 452:488] = BLUE
        self.assertIsNone(tracker([(RACK, 0.061)]).detect(rgb))
        rgb[532:568, 452:488] = BLUE
        self.assertEqual(tracker([(RACK, 0.061)]).detect(rgb), RACK)

    def test_the_rack_prior_sits_on_the_new_desk_rack(self):
        # 11-17: before the lock the window waits at the prior; the new-desk rack's following centres span
        # x 243-312, y 561-583 (the old-desk (384, 578) left the window on the rack's right half)
        cx, cy = fv.PRIOR_CENTRE["rack"]
        self.assertTrue(243 <= cx <= 312 and 561 <= cy <= 583, (cx, cy))

    def test_round_holes_count(self):
        rgb = np.full((720, 1280, 3), 200, np.uint8)
        import cv2
        for cx in (300, 360, 420):
            cv2.circle(rgb, (cx, 600), 22, BLUE, -1)     # ~1500 px discs, like the new-desk holes
        self.assertEqual(len(fv.rack_holes(rgb)), 3)

    def test_a_rack_box_with_its_holes_covered_is_not_taken(self):
        self.assertIsNone(tracker([(RACK, 0.061)]).detect(scene(holes=False)))


if __name__ == "__main__":
    unittest.main()
