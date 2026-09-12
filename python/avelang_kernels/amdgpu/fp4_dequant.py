"""Device-side FP4 dequantization helpers for AMDGPU."""

import avelang
import avelang.language as al

FP4_MASK = 0x8E8E8E8E


@avelang.jit
def fp4_dequant(
    q: al.u32,
    scale: al.f32,
    out: al.Tensor((1, 4), al.u32),
):
    scale_pair = al.make_local((1, 2), al.f32)
    scale_pair[0, 0] = scale
    scale_pair[0, 1] = scale

    mask = al.convert(FP4_MASK, al.u32)
    lo = q & mask
    hi = al.amdgpu.bitreverse(q) & mask
    f0 = al.amdgpu.cvt_pk_f32_bf8(lo, 0) * scale_pair[0]
    f1 = al.amdgpu.cvt_pk_f32_bf8(lo, 1) * scale_pair[0]
    f2 = al.amdgpu.cvt_pk_f32_bf8(hi, 0) * scale_pair[0]
    f3 = al.amdgpu.cvt_pk_f32_bf8(hi, 1) * scale_pair[0]
    out[0, 0] = al.amdgpu.perm(
        al.bitcast(f1[1], al.u32), al.bitcast(f0[1], al.u32), 0x07060302
    )
    out[0, 1] = al.amdgpu.perm(
        al.bitcast(f1[0], al.u32), al.bitcast(f0[0], al.u32), 0x07060302
    )
    out[0, 2] = al.amdgpu.perm(
        al.bitcast(f3[1], al.u32), al.bitcast(f2[1], al.u32), 0x07060302
    )
    out[0, 3] = al.amdgpu.perm(
        al.bitcast(f3[0], al.u32), al.bitcast(f2[0], al.u32), 0x07060302
    )
