#!/usr/bin/env python3
import unittest

import torch

import avelang
import avelang.language as S
from avelang.testing import has_rocm


@avelang.jit
def kernel_bitreverse(
    value: S.Tensor((1,), S.u32),
    result: S.Tensor((1,), S.u32),
):
    result[0] = S.amdgpu.bitreverse(value[0])


@unittest.skipUnless(has_rocm(), "Requires a ROCm GPU.")
class TestAMDGPUBitOperations(unittest.TestCase):
    def test_bitreverse(self):
        value = 0x01234567
        src = torch.tensor([value], dtype=torch.uint32, device="cuda")
        result = torch.empty_like(src)

        kernel_bitreverse[lambda: ((1, 1, 1), (1, 1, 1))](src, result)

        expected = int(f"{value:032b}"[::-1], 2)
        self.assertEqual(result.item(), expected)


if __name__ == "__main__":
    unittest.main()
