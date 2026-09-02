import unittest

import torch

from starVLA.model.framework.VLM4A.CosmosGR00T_N1d7 import CosmosGR00T_N1d7


class CosmosGR00TActionLossMaskTest(unittest.TestCase):
    def test_left_side_dimensions_have_zero_loss_weight(self):
        framework = object.__new__(CosmosGR00T_N1d7)
        framework.action_dim = 54
        framework.max_action_dim = 132
        framework.loss_action_dim_range = (27, 54)
        mask = torch.ones((2, 30, 132), dtype=torch.float32)

        masked = framework._apply_loss_action_dim_range(mask)

        self.assertEqual(float(masked[:, :, :27].sum()), 0.0)
        self.assertEqual(float(masked[:, :, 27:54].min()), 1.0)
        self.assertEqual(float(masked[:, :, 54:].sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
