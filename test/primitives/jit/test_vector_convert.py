"""Explicit numeric vector conversions preserve shape and rounding."""

import avelang
import avelang.language as al
import pytest
import torch
from avelang.testing import has_rocm


@avelang.jit
def narrow_pairs(src: al.Tensor((64, 2), al.f32), dst: al.Tensor((64, 2), al.bf16)):
    lane = al.thread_id(0)
    dst[lane] = al.convert(src[lane], al.bf16)


@avelang.jit
def widen_pairs(src: al.Tensor((64, 2), al.bf16), dst: al.Tensor((64, 2), al.f32)):
    lane = al.thread_id(0)
    dst[lane] = al.convert(src[lane], al.f32)


@avelang.jit
def unsigned_pairs(src: al.Tensor((64, 2), al.u32), dst: al.Tensor((64, 2), al.f64)):
    lane = al.thread_id(0)
    dst[lane] = al.convert(src[lane], al.f64)


@avelang.jit
def same_width_pairs(src: al.Tensor((64, 2), al.f16), dst: al.Tensor((64, 2), al.bf16)):
    lane = al.thread_id(0)
    dst[lane] = al.convert(src[lane], al.bf16)


def test_conversion_keeps_vector_operations_in_ir():
    import _avelang_bindings as bindings
    from avelang.compiler.code_generator import _build_import_module, _get_function_def

    generator = bindings.MLIRGenerator()
    generator.generate_from_python_ast(_build_import_module([narrow_pairs]))
    generator.visit_function_def(_get_function_def(narrow_pairs.parse()), "[]", "kernel")
    ir = generator.get_mlir()
    assert "arith.truncf" in ir
    assert "vector<2xf32> to vector<2xbf16>" in ir


@pytest.mark.skipif(not has_rocm(), reason="Requires AMD GPU")
def test_vector_bf16_rounding_and_widening():
    # Include both sides of BF16 midpoints, ties with even/odd low bits,
    # negative values, signed zero, subnormals and infinities.
    bits = torch.tensor(
        [
            0x3F807FFF,
            0x3F808000,
            0x3F808001,
            0x3F818000,
            0xBF808000,
            0xBF818000,
            0,
            0x80000000,
            0x00008000,
            0x00018000,
            0x7F800000,
            0xFF800000,
            0x3F000000,
            0xBF000000,
            0x477FE000,
            0xC77FE000,
        ],
        dtype=torch.uint32,
    )
    src = bits.repeat(8).view(torch.float32).reshape(64, 2).cuda()
    dst = torch.empty_like(src, dtype=torch.bfloat16)
    narrow_pairs[lambda: ((1, 1, 1), (64, 1, 1))](src, dst)
    torch.testing.assert_close(dst.view(torch.int16), src.bfloat16().view(torch.int16), rtol=0, atol=0)
    wide = torch.empty_like(src)
    widen_pairs[lambda: ((1, 1, 1), (64, 1, 1))](dst, wide)
    torch.testing.assert_close(wide.view(torch.int32), dst.float().view(torch.int32), rtol=0, atol=0)


@pytest.mark.skipif(not has_rocm(), reason="Requires AMD GPU")
def test_vector_conversion_preserves_unsigned_values():
    src = torch.tensor([0, 1, 0x80000000, 0xFFFFFFFF], dtype=torch.uint32).repeat(32).reshape(64, 2).cuda()
    dst = torch.empty((64, 2), dtype=torch.float64, device="cuda")
    unsigned_pairs[lambda: ((1, 1, 1), (64, 1, 1))](src, dst)
    torch.testing.assert_close(dst, src.double(), rtol=0, atol=0)


@pytest.mark.skipif(not has_rocm(), reason="Requires AMD GPU")
def test_vector_conversion_between_float_formats():
    src = torch.linspace(-100, 100, 128, device="cuda", dtype=torch.float16).reshape(64, 2)
    dst = torch.empty_like(src, dtype=torch.bfloat16)
    same_width_pairs[lambda: ((1, 1, 1), (64, 1, 1))](src, dst)
    torch.testing.assert_close(dst, src.bfloat16(), rtol=0, atol=0)
