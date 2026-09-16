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


@avelang.jit
def kernel_unsigned_shift_after_make_tensor(src: S.Pointer(S.u32), out: S.Tensor((64,), S.u32)):
    lane = S.thread_id(0)
    values = S.make_tensor(src, S.u32, S.make_layout((64,), (1,)))
    out[lane] = values[lane] >> 24


@unittest.skipUnless(has_rocm(), "Requires a ROCm GPU.")
class TestAMDGPUBitOperations(unittest.TestCase):
    def test_bitreverse(self):
        value = 0x01234567
        src = torch.tensor([value], dtype=torch.uint32, device="cuda")
        result = torch.empty_like(src)

        kernel_bitreverse[lambda: ((1, 1, 1), (1, 1, 1))](src, result)

        expected = int(f"{value:032b}"[::-1], 2)
        self.assertEqual(result.item(), expected)

    def test_unsigned_shift_after_make_tensor(self):
        src = torch.full((64,), 0xFF000001, dtype=torch.uint32, device="cuda")
        out = torch.empty_like(src)
        kernel_unsigned_shift_after_make_tensor[lambda: ((1, 1, 1), (64, 1, 1))](src, out)
        torch.testing.assert_close(out, torch.full_like(src, 255), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
