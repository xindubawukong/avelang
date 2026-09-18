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
    scale = al.bitcast(exponent << 23, al.f32)
    packed = al.amdgpu.cvt_scalef32_pk_fp4_f32(al.convert(0, al.u32), act[0], act[1], scale, 0)
    packed = al.amdgpu.cvt_scalef32_pk_fp4_f32(packed, act[2], act[3], scale, 1)
    return packed, exponent
