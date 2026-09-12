#!/usr/bin/env python3
import unittest

import torch
from avelang.testing import has_rocm
from avelang_kernels.amdgpu import fp4_gemm
from avelang_kernels.amdgpu.fp4_gemm_utils import (
    dequantize_fp4,
    packed_scale_shape,
    packed_weight_shape,
    process_fp4_scales,
    repack_fp4,
)


class TestAMDGPUFP4Format(unittest.TestCase):
    def test_packed_shapes_and_dtypes(self):
        n, k = 64, 256
        torch.manual_seed(11)
        qweight = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8)
        scales = torch.ones((n, k // 16), dtype=torch.float8_e4m3fn)

        packed_weight = repack_fp4(qweight)
        packed_scales = process_fp4_scales(scales)

        self.assertEqual(packed_weight.shape, packed_weight_shape(n, k))
        self.assertEqual(packed_weight.dtype, torch.int32)
        self.assertTrue(packed_weight.is_contiguous())
        self.assertEqual(packed_scales.shape, packed_scale_shape(n, k))
        self.assertEqual(packed_scales.dtype, torch.uint8)
        self.assertTrue(packed_scales.is_contiguous())


@unittest.skipUnless(has_rocm(), "Requires ROCm")
class TestAMDGPUFP4GEMM(unittest.TestCase):
    def test_matches_torch_reference(self):
        cases = (
            (16, 64, 256, 17, None),
            (64, 128, 256, 1234, None),
            (96, 64, 512, 2026, None),
            (256, 256, 256, 31, None),
            (160, 256, 256, 41, 0x12211101100A),
            (224, 192, 256, 43, 0x122111010C0E),
            (224, 256, 256, 47, 0x12211101100E),
        )
        for m, n, k, seed, solution_id in cases:
            with self.subTest(m=m, n=n, k=k, solution_id=solution_id):
                torch.manual_seed(seed)
                a = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
                qweight = torch.randint(
                    0, 256, (n, k // 2), dtype=torch.uint8, device="cuda"
                )
                scales = (torch.rand((n, k // 16), device="cuda") * 3.5 + 0.25).to(
                    torch.float8_e4m3fn
                )
                global_scale = 0.25

                packed_weight = repack_fp4(qweight)
                packed_scales = process_fp4_scales(scales)
                actual = fp4_gemm(
                    a,
                    packed_weight,
                    packed_scales,
                    global_scale,
                    solution_id=solution_id,
                )
                expected = (
                    a.float() @ (dequantize_fp4(qweight, scales) * global_scale).T
                ).to(torch.bfloat16)

                torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)

    def test_rejects_source_format(self):
        a = torch.zeros((16, 256), dtype=torch.bfloat16, device="cuda")
        qweight = torch.zeros((64, 128), dtype=torch.uint8, device="cuda")
        scales = torch.ones((64, 16), dtype=torch.float8_e4m3fn, device="cuda")
        with self.assertRaisesRegex(ValueError, "AveLang packed"):
            fp4_gemm(a, qweight, scales, 1.0)


if __name__ == "__main__":
    unittest.main()
