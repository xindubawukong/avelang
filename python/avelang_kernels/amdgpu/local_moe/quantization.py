"""MXFP4 quantization fused into the Stage1 intermediate store."""

import avelang
import avelang.language as al


@avelang.jit
def quantize_mxfp4_activation(act: al.Tensor((4,), al.f32)) -> (al.u32, al.u32):
    """Eight neighboring lanes quantize one 32-value block (four per lane)."""
    maximum = al.amdgpu.maximum_f32(
        al.amdgpu.maximum_f32(al.abs(act[0]), al.abs(act[1])),
        al.amdgpu.maximum_f32(al.abs(act[2]), al.abs(act[3])),
    )
    peer = al.bitcast(al.amdgpu.ds_swizzle(al.bitcast(maximum, al.u32), 0x041F), al.f32)
    maximum = al.amdgpu.maximum_f32(maximum, peer)
    peer = al.bitcast(al.amdgpu.ds_swizzle(al.bitcast(maximum, al.u32), 0x081F), al.f32)
    maximum = al.amdgpu.maximum_f32(maximum, peer)
    peer = al.bitcast(al.amdgpu.ds_swizzle(al.bitcast(maximum, al.u32), 0x101F), al.f32)
    maximum = al.amdgpu.maximum_f32(maximum, peer)
    # Petit's AiterMxFp4Quantization uses different scale rounding from the
    # input preprocessor's RoundUp convention.
    bits = (al.bitcast(maximum, al.u32) + 0x00400000) & al.convert(0xFF800000, al.u32)
    exponent = al.max(bits >> 23, 2) - 2
    inverse_scale = al.bitcast((254 - exponent) << 23, al.f32)
    packed = al.convert(0, al.u32)
    for c in al.static_range(4):
        bits = al.bitcast(act[c] * inverse_scale, al.u32)
        magnitude = bits & 0x7FFFFFFF
        rounded = al.bitcast(al.bitcast(magnitude, al.f32) + al.convert(4194304.0, al.f32), al.u32) - 0x4A800000
        normal = (magnitude + al.convert(0xC11FFFFF, al.u32) + ((magnitude >> 22) & 1)) >> 22
        code = al.select(magnitude < 0x40C00000, normal, al.convert(7, al.u32))
        code = al.select(magnitude < 0x3F800000, rounded, code)
        code = code | ((bits >> 28) & 8)
        packed = packed | (code << (c * 4))
    return packed, exponent
