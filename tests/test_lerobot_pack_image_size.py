import unittest
from unittest import mock

import numpy as np
from PIL import Image

from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset


class LeRobotPackImageSizeTest(unittest.TestCase):
    @staticmethod
    def _dataset(obs_image_size):
        dataset = object.__new__(LeRobotSingleDataset)
        dataset.data_cfg = {"obs_image_size": obs_image_size, "include_state": False}
        dataset._modality_keys = {
            "video": ["video.ego", "video.wrist"],
            "language": ["annotation.task"],
            "action": ["action.arm"],
        }
        dataset.tag = "new_embodiment"
        return dataset

    @staticmethod
    def _sample(height, width):
        return {
            "video.ego": np.zeros((1, height, width, 3), dtype=np.uint8),
            "video.wrist": np.zeros((1, height, width, 3), dtype=np.uint8),
            "annotation.task": ["test instruction"],
            "action.arm": np.zeros((30, 7), dtype=np.float32),
        }

    def test_matching_images_are_not_resized_again(self):
        dataset = self._dataset([256, 256])

        with mock.patch.object(
            Image.Image,
            "resize",
            side_effect=AssertionError("matching images must not be resized twice"),
        ):
            packed = dataset._pack_sample(self._sample(256, 256))

        self.assertEqual([image.size for image in packed["image"]], [(256, 256), (256, 256)])

    def test_config_uses_height_width_order(self):
        dataset = self._dataset([192, 320])
        packed = dataset._pack_sample(self._sample(64, 96))

        self.assertEqual([image.size for image in packed["image"]], [(320, 192), (320, 192)])


if __name__ == "__main__":
    unittest.main()
