"""Run with the serving Python: unittest, CPU only, no weights/camera/robot.

The focus tap's idle preview: with no request for TAP_IDLE_S the tracker thread offers its own
frame and what a request would crop from it (newest record, else the prior), marked preview.
"""
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deployment.model_server import focus_view as fv

SENTENCE = "Aim the pipette held in the right hand at the tube held in the left hand."


class Tracker:
    def __init__(self, **kwargs):
        self.locked = False

    def reset(self, target):
        pass

    def step(self, frame):
        return dict(target="tube", window=[100., 50., 400., 350.], center=[250., 200.], following=1,
                    visibility=0.9, locked=True, acquired=False, box=None, ms=1.)

    crop = staticmethod(fv.FocusTracker.crop)


class Tap:
    def __init__(self):
        self.offers = []

    def offer(self, ego, crop, rec):
        self.offers.append((crop, rec))


class TapPreviewTest(unittest.TestCase):
    def setUp(self):
        with mock.patch.object(fv, "FocusTracker", Tracker):
            self.synth = fv.FocusViewSynth(device="cpu")      # no camera: no thread, no tap socket
        self.synth._tap = self.tap = Tap()
        self.ego = np.random.default_rng(0).integers(0, 256, (720, 1280, 3), dtype=np.uint8)

    def test_nothing_before_the_first_request(self):
        self.synth._tap_preview(self.ego)
        self.assertEqual(self.tap.offers, [])

    def test_requests_own_the_tiles_then_idle_preview_takes_over(self):
        crop, rec = self.synth(self.ego, SENTENCE)
        self.assertEqual(len(self.tap.offers), 1)
        self.assertNotIn("preview", self.tap.offers[0][1])
        self.synth._tap_preview(self.ego)                     # inside TAP_IDLE_S: suppressed
        self.assertEqual(len(self.tap.offers), 1)
        self.synth._tap_offered -= fv.TAP_IDLE_S
        self.synth._tap_preview(self.ego)
        crop2, rec2 = self.tap.offers[-1]
        self.assertTrue(rec2["preview"])
        self.assertEqual(rec2["window"], rec["window"])
        np.testing.assert_array_equal(crop2, crop)
        self.assertNotIn("preview", rec)                     # the served record is untouched

    def test_after_reset_the_preview_is_the_prior_and_tracking_does_not_restart(self):
        self.synth(self.ego, SENTENCE)
        self.synth.reset()
        self.synth._tap_offered -= fv.TAP_IDLE_S
        self.synth._tap_preview(self.ego)
        rec = self.tap.offers[-1][1]
        self.assertEqual(rec["window"], self.synth._prior_rec(self.ego.shape, "tube")["window"])
        self.assertTrue(rec["preview"])
        self.assertIsNone(self.synth._sentence)

    def test_window_view_marks_the_preview(self):
        rec = dict(Tracker().step(None), preview=True)
        img = fv.FocusTap.window_view(self.ego, rec)
        self.assertEqual(img.shape, (360, 640, 3))


if __name__ == "__main__":
    unittest.main()
