"""Run with the serving Python: unittest, CPU only, no downloads/camera/robot.

Uses the installed Transformers SAM2 session and its REAL memory selectors.
No model weights are loaded. A fake forward tests the FocusTracker call site.
"""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deployment.model_server.focus_history import prune_streaming_history
from deployment.model_server.focus_view import FocusTracker
from transformers.models.sam2_video.modeling_sam2_video import Sam2VideoInferenceSession, Sam2VideoModel


def new_session():
    s = Sam2VideoInferenceSession(dtype=torch.float32)
    s.obj_id_to_idx(1)
    return s


def add_frame(s, i):
    s.add_new_frame(torch.zeros(3, 2, 2), frame_idx=i)
    output = dict(marker=i, object_pointer=torch.tensor([float(i)]))
    s.store_output(0, i, output_value=output, is_conditioning_frame=(i == 0))
    s.frames_tracked_per_obj[0][i] = {"reverse": False}


class HistoryTest(unittest.TestCase):
    def test_real_memory_selection_identical_and_bounded_for_3000_frames(self):
        full, bounded = new_session(), new_session()
        model = SimpleNamespace(num_maskmem=7, config=SimpleNamespace(max_object_pointers_in_encoder=16),
                                training=False)
        for i in range(3000):
            add_frame(full, i)
            add_frame(bounded, i)
            stats = prune_streaming_history(bounded, i, num_maskmem=7, max_object_pointers=16)
            self.assertLessEqual(stats['raw_frames'], 2)
            self.assertLessEqual(stats['recent_outputs'], 16)
            self.assertLessEqual(len(bounded.frames_tracked_per_obj[0]), 17)
            self.assertEqual(stats['conditioning_outputs'], 1)
            # The next streaming forward sees exactly the same tensors and
            # temporal positions, despite dropping historical image storage.
            a = Sam2VideoModel._gather_memory_frame_outputs(model, full, 0, i+1)
            b = Sam2VideoModel._gather_memory_frame_outputs(model, bounded, 0, i+1)
            simplify = lambda rows: [(t, None if o is None else o['marker']) for t, o in rows]
            self.assertEqual(simplify(a), simplify(b))
            a = Sam2VideoModel._get_object_pointers(model, full, 0, i+1, full.num_frames, 'cpu', streaming=True)
            b = Sam2VideoModel._get_object_pointers(model, bounded, 0, i+1, bounded.num_frames, 'cpu', streaming=True)
            self.assertEqual(a[0], b[0])
            self.assertEqual(a[2], b[2])
            for x, y in zip(a[1], b[1]):
                torch.testing.assert_close(x, y, rtol=0, atol=0)

    def test_tracker_passes_monotonic_indices_and_resets_on_reacquire(self):
        calls = []

        class Model:
            num_maskmem = 7
            config = SimpleNamespace(max_object_pointers_in_encoder=16)

            def __call__(self, *, inference_session, frame, frame_idx):
                calls.append(frame_idx)
                add_frame(inference_session, frame_idx)
                return SimpleNamespace(pred_masks=torch.ones(1,1,2,2))

        class Processor:
            def __call__(self, **kwargs):
                return SimpleNamespace(pixel_values=torch.zeros(1,3,2,2), original_sizes=[(2,2)])

            def init_video_session(self, **kwargs):
                return new_session()

            def add_inputs_to_inference_session(self, **kwargs):
                pass

            def post_process_masks(self, masks, **kwargs):
                return masks

        tracker = FocusTracker.__new__(FocusTracker)
        tracker.dev, tracker.sam, tracker.sam_p = 'cpu', Model(), Processor()
        tracker.reset('holder')
        rgb = np.zeros((2,2,3), np.uint8)
        for i in range(100):
            self.assertEqual(tracker._sam(rgb, box=[0,0,1,1] if i == 0 else None), [0,0,1,1,4])
        self.assertEqual(calls, list(range(100)))
        self.assertEqual(tracker.sam_cache['frame_idx'], 99)
        self.assertEqual(tracker.sam_cache['raw_frames'], 2)
        self.assertEqual(tracker.sam_cache['recent_outputs'], 16)
        tracker.reset('rack')
        self.assertIsNone(tracker.sam_cache)
        self.assertIsNone(tracker.sess)
        tracker._sam(rgb, box=[0,0,1,1])
        self.assertEqual(calls[-1], 0)


if __name__ == '__main__':
    unittest.main()
