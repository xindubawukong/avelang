#!/usr/bin/env python3
import unittest
from dataclasses import replace

import torch
from avelang.testing import has_rocm
from avelang_kernels.amdgpu import fp4_gemm
from avelang_kernels.amdgpu.fp4_gemm.config import FP4GemmConfig
from avelang_kernels.amdgpu.fp4_gemm.memory_ops import FP4MemoryConfig
from avelang_kernels.amdgpu.fp4_gemm.solution import (
    MatmulElementB, MatmulFeatures, MatmulMfmaType, SolutionId,
)
from avelang_kernels.amdgpu.fp4_gemm.utils import (
    dequantize_fp4,
    packed_scale_shape,
    packed_weight_shape,
    process_fp4_scales,
    repack_fp4,
)

FP4_VARIANTS = (
    (torch.float16, MatmulMfmaType.FP16, MatmulElementB.NVFP4),
    (torch.bfloat16, MatmulMfmaType.BF16, MatmulElementB.NVFP4),
    (torch.bfloat16, MatmulMfmaType.BF16, MatmulElementB.MXFP4),
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

    def test_mxfp4_scales(self):
        qweight = torch.full((64, 128), 0x22, dtype=torch.uint8)  # FP4 = 1
        scales = torch.full((64, 8), 127, dtype=torch.uint8)
        scales[:, 0] = 0
        scales[:, 1] = 254
        scales[:, 2] = 255
        packed = process_fp4_scales(scales, element_b=MatmulElementB.MXFP4)
        self.assertEqual(packed.shape, (4, 64, 2))
        self.assertEqual(packed.dtype, torch.uint8)
        self.assertTrue(packed.is_contiguous())
        weight = dequantize_fp4(qweight, scales, element_b=MatmulElementB.MXFP4)
        self.assertTrue(torch.all(weight[:, :32] == 2.0**-127))
        self.assertTrue(torch.all(weight[:, 32:64] == 2.0**127))
        self.assertTrue(torch.isnan(weight[:, 64:96]).all())
        self.assertTrue(torch.all(weight[:, 96:] == 1.0))


@unittest.skipUnless(has_rocm(), "Requires ROCm")
class TestAMDGPUFP4GEMM(unittest.TestCase):
    def _check_random_gemm(
        self, m, n, k, seed, solution_id,
        dtype=torch.bfloat16, element_b=MatmulElementB.NVFP4, high_precision=None,
    ):
        torch.manual_seed(seed)
        a = torch.randn((m, k), dtype=dtype, device="cuda")
        qweight = torch.randint(
            0, 256, (n, k // 2), dtype=torch.uint8, device="cuda"
        )
        if element_b == MatmulElementB.MXFP4:
            scales = torch.randint(124, 131, (n, k // 32), dtype=torch.uint8, device="cuda")
        else:
            scales = (torch.rand((n, k // 16), device="cuda") * 3.5 + 0.25).to(
                torch.float8_e4m3fn
            )
        packed_weight = repack_fp4(qweight)
        packed_scales = process_fp4_scales(scales, element_b=element_b)
        global_scale = 0.25

        # Guard the M tail and use NaNs to catch unwritten output elements.
        guard_size = 32
        storage = torch.full(
            (m * n + 2 * guard_size,), 123.0,
            dtype=dtype, device="cuda",
        )
        out = storage[guard_size:-guard_size].view(m, n)
        out.fill_(float("nan"))
        actual = fp4_gemm(
            a, packed_weight, packed_scales, global_scale,
            out=out, solution_id=solution_id,
            element_b=element_b, high_precision=high_precision,
        )
        expected = (
            a.float() @ (dequantize_fp4(qweight, scales, element_b=element_b) * global_scale).T
        ).to(dtype)
        tolerance = 2e-3 if dtype == torch.float16 else 2e-2
        torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
        for guard in (storage[:guard_size], storage[-guard_size:]):
            torch.testing.assert_close(
                guard, torch.full_like(guard, 123.0), rtol=0, atol=0
            )

    def test_matches_torch_reference(self):
        cases = (
            (16, 64, 256, 17, None),
            (64, 128, 256, 1234, None),
            (96, 64, 512, 2026, None),
            (256, 256, 256, 31, None),
            (200, 128, 256, 37, 0x122111020408),
            (80, 192, 256, 39, 0x212111020606),
            (160, 256, 256, 41, 0x12211101100A),
            (224, 192, 256, 43, 0x122111010C0E),
            (224, 256, 256, 47, 0x12211101100E),
        )
        for m, n, k, seed, solution_id in cases:
            with self.subTest(m=m, n=n, k=k, solution_id=solution_id):
                self._check_random_gemm(m, n, k, seed, solution_id)

    def test_pipeline_tail(self):
        # Both use GROUP_K=256, so all five tile counts fit the packed format.
        for solution_id, single_buffer in (
            (0x122111040402, False),
            (0x122111040404, True),
        ):
            config = FP4GemmConfig.from_solution(SolutionId.from_int(solution_id))
            memory = FP4MemoryConfig.from_config(config)
            self.assertEqual(memory.single_buffer, single_buffer)
            for k_total in (1, 2, 3, 4, 5):
                with self.subTest(single_buffer=single_buffer, k_total=k_total):
                    self._check_random_gemm(
                        config.group_m + 1, 2 * config.group_n,
                        k_total * config.group_k, 100 + k_total, solution_id,
                    )

    def test_type_and_precision_variants(self):
        for dtype, mfma, element_b in FP4_VARIANTS:
            for high_precision in (False, True):
                with self.subTest(dtype=dtype, format=element_b, high_precision=high_precision):
                    self._check_random_gemm(
                        17, 128, 512, 73, None, dtype, element_b, high_precision,
                    )
                # K-warp 1/2/4, both LDS buffering modes, and a partial M block.
                for base_id in (0x122111040402, 0x212111040204, 0x411111040202):
                    features = MatmulFeatures.GRID
                    if high_precision:
                        features |= MatmulFeatures.HIGH_PRECISION
                    solution = replace(
                        SolutionId.from_int(base_id), element_b=element_b,
                        mfma_type=mfma, features=features,
                    )
                    with self.subTest(solution_id=hex(int(solution))):
                        self._check_random_gemm(
                            solution.group_m + 1, 2 * solution.group_n,
                            5 * solution.group_k, 79, solution,
                            dtype, element_b,  # Precision comes from solution_id.
                        )

    def test_high_precision_preserves_small_weights(self):
        qweight = torch.full((64, 128), 0x11, dtype=torch.uint8, device="cuda")
        weight = repack_fp4(qweight)
        for dtype, element_b, source_scale, a_value, global_scale in (
            (torch.float16, MatmulElementB.NVFP4, 2.0**-9, 1.0, 1.0),
            (torch.bfloat16, MatmulElementB.MXFP4, 7, 2.0**60, 2.0**60),
        ):
            with self.subTest(dtype=dtype, format=element_b):
                is_mx = element_b == MatmulElementB.MXFP4
                a = torch.full((1, 256), a_value, dtype=dtype, device="cuda")
                scales = torch.full(
                    (64, 8 if is_mx else 16), source_scale,
                    dtype=torch.uint8 if is_mx else torch.float8_e4m3fn, device="cuda",
                )
                packed = process_fp4_scales(scales, element_b=element_b)
                actual = fp4_gemm(a, weight, packed, global_scale, element_b=element_b, high_precision=True)
                expected = torch.full_like(actual, 128.0 if is_mx else 0.25)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_pipeline_lds_reuse(self):
        # In this single-buffer config, writeback overlaps A's last K atom
        # in LDS. Exercise many blocks and K iterations before reusing it.
        for seed in (83, 89, 97):
            with self.subTest(seed=seed):
                self._check_random_gemm(
                    1024, 1024, 16384, seed, 0x122011040404, torch.float16,
                )

    def test_mxfp4_high_precision_large_scales(self):
        a = torch.full((1, 256), 2.0**-120, dtype=torch.bfloat16, device="cuda")
        qweight = torch.full((64, 128), 0x11, dtype=torch.uint8, device="cuda")
        scales = torch.full((64, 8), 248, dtype=torch.uint8, device="cuda")
        actual = fp4_gemm(
            a, repack_fp4(qweight), process_fp4_scales(scales, element_b=MatmulElementB.MXFP4),
            1.0, element_b=MatmulElementB.MXFP4, high_precision=True,
        )
        torch.testing.assert_close(actual, torch.full_like(actual, 256.0), rtol=0, atol=0)

    def test_rejects_incompatible_variants(self):
        a = torch.ones((16, 256), dtype=torch.float16, device="cuda")
        qweight = repack_fp4(torch.zeros((64, 128), dtype=torch.uint8, device="cuda"))
        scales = process_fp4_scales(torch.ones((64, 16), dtype=torch.float8_e4m3fn, device="cuda"))
        with self.assertRaisesRegex(ValueError, "MXFP4 requires BF16"):
            fp4_gemm(a, qweight, scales, 1.0, element_b=MatmulElementB.MXFP4)
        with self.assertRaisesRegex(ValueError, "input dtype"):
            fp4_gemm(a, qweight, scales, 1.0, solution_id=0x122111040402)
        with self.assertRaisesRegex(ValueError, "high_precision"):
            fp4_gemm(a, qweight, scales, 1.0, solution_id=0x122011040402, high_precision=True)
        with self.assertRaisesRegex(ValueError, "AveLang packed"):
            fp4_gemm(a.bfloat16(), qweight, scales, 1.0, element_b=MatmulElementB.MXFP4)

    def test_small_m_and_partial_blocks(self):
        for solution_id, warp_k in (
            (0x122111020804, 1),
            (0x212111040204, 2),
            (0x411111040202, 4),
        ):
            config = FP4GemmConfig.from_solution(SolutionId.from_int(solution_id))
            self.assertEqual(config.warp_k, warp_k)
            for m in (1, 15, 17, config.group_m - 1, config.group_m + 1):
                with self.subTest(warp_k=warp_k, m=m):
                    self._check_random_gemm(
                        m, 2 * config.group_n, 512, 200 + m, solution_id,
                    )

    def test_fragment_mapping(self):
        n = k = 256
        row = torch.arange(n, device="cuda")[:, None]
        col = torch.arange(k, device="cuda")[None, :]
        # Distinguish lanes within a fragment as well as neighboring K/N tiles.
        codes = ((3 * row + 5 * col + row // 16 + 7 * (col // 16)) % 16).to(
            torch.uint8
        )
        qweight = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()
        scale_values = torch.tensor((0.5, 1, 2, 4, 8), device="cuda")
        packed_weight = repack_fp4(qweight)

        # I @ W.T exposes every dequantized value, independently of packed layout.
        values = torch.tensor(
            (0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6),
            dtype=torch.float32, device="cuda",
        )
        for dtype, mfma, element_b in FP4_VARIANTS:
            a = torch.eye(k, dtype=dtype, device="cuda")
            group_size = 32 if element_b == MatmulElementB.MXFP4 else 16
            scale_col = torch.arange(k // group_size, device="cuda")[None, :]
            numeric_scales = scale_values[(row // 16 + 3 * scale_col) % 5]
            scales = (
                (numeric_scales.log2() + 127).to(torch.uint8)
                if element_b == MatmulElementB.MXFP4
                else numeric_scales.to(torch.float8_e4m3fn)
            )
            packed_scales = process_fp4_scales(scales, element_b=element_b)
            weight = values[codes.long()] * numeric_scales.repeat_interleave(group_size, dim=1)
            expected = (weight.T * 0.25).to(dtype)
            for high_precision in (False, True):
                # Multiple N atoms, both buffering modes, and K-warp reduction.
                for base_id in (0x12211101100A, 0x12211101100E, 0x411111040202):
                    solution = replace(
                        SolutionId.from_int(base_id), element_b=element_b, mfma_type=mfma,
                        features=MatmulFeatures.GRID | (MatmulFeatures.HIGH_PRECISION if high_precision else 0),
                    )
                    with self.subTest(solution_id=hex(int(solution))):
                        actual = fp4_gemm(
                            a, packed_weight, packed_scales, 0.25,
                            solution_id=solution, element_b=element_b,
                        )
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_global_scale_tensor_is_graph_safe(self):
        for dtype, _, element_b in FP4_VARIANTS:
            for high_precision in (False, True):
                with self.subTest(dtype=dtype, format=element_b, high_precision=high_precision):
                    self._check_graph_scale(dtype, element_b, high_precision)

    def _check_graph_scale(self, dtype, element_b, high_precision):
        m, n, k = 16, 64, 256
        torch.manual_seed(53)
        a = torch.randn((m, k), dtype=dtype, device="cuda")
        qweight = torch.randint(
            0, 256, (n, k // 2), dtype=torch.uint8, device="cuda"
        )
        if element_b == MatmulElementB.MXFP4:
            scales = torch.full((n, k // 32), 127, dtype=torch.uint8, device="cuda")
        else:
            scales = torch.ones((n, k // 16), dtype=torch.float8_e4m3fn, device="cuda")
        packed_weight = repack_fp4(qweight)
        packed_scales = process_fp4_scales(scales, element_b=element_b)
        weight = dequantize_fp4(qweight, scales, element_b=element_b)
        global_scale = torch.tensor([0.25], dtype=torch.float32, device="cuda")
        output = torch.empty((m, n), dtype=dtype, device="cuda")

        fp4_gemm(
            a, packed_weight, packed_scales, global_scale, out=output,
            element_b=element_b, high_precision=high_precision,
        )
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fp4_gemm(
                a, packed_weight, packed_scales, global_scale, out=output,
                element_b=element_b, high_precision=high_precision,
            )

        for scale in (0.25, 0.5):
            global_scale.fill_(scale)
            graph.replay()
            torch.cuda.synchronize()
            expected = (a.float() @ (weight * scale).T).to(dtype)
            torch.testing.assert_close(
                output, expected, rtol=2e-2, atol=2e-2
            )

    def test_rejects_source_format(self):
        a = torch.zeros((16, 256), dtype=torch.bfloat16, device="cuda")
        qweight = torch.zeros((64, 128), dtype=torch.uint8, device="cuda")
        scales = torch.ones((64, 16), dtype=torch.float8_e4m3fn, device="cuda")
        with self.assertRaisesRegex(ValueError, "AveLang packed"):
            fp4_gemm(a, qweight, scales, 1.0)


if __name__ == "__main__":
    unittest.main()
