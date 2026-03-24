import unittest

import torch

from scripts.smooth_policy.amp_history.amp_training.td3.helper.td3_checkpointing import (
    _to_rng_byte_tensor,
)


class RngStateConversionTests(unittest.TestCase):
    def test_to_rng_byte_tensor_from_int_tensor(self) -> None:
        value = torch.tensor([0, 1, 255], dtype=torch.int64)
        converted = _to_rng_byte_tensor(value)
        self.assertEqual(converted.dtype, torch.uint8)
        self.assertTrue(torch.equal(converted, torch.tensor([0, 1, 255], dtype=torch.uint8)))

    def test_to_rng_byte_tensor_from_python_list(self) -> None:
        converted = _to_rng_byte_tensor([0, 2, 4, 8])
        self.assertEqual(converted.dtype, torch.uint8)
        self.assertTrue(torch.equal(converted, torch.tensor([0, 2, 4, 8], dtype=torch.uint8)))


if __name__ == "__main__":
    unittest.main()
