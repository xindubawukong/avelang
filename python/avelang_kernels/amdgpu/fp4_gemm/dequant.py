"""Device-side FP4 dequantization helpers for AMDGPU."""

from __future__ import annotations

import avelang
import avelang.language as al

FP4_MASK = 0x8E8E8E8E


@avelang.jit
def dequant_scales(
    config: al.constexpr,
    packed_scale: al.u16,
    out: al.Tensor((1, 2), al.f32),
):
    packed = al.convert(packed_scale, al.u32)
    bits = al.convert(0, al.u32)
    if config.gemm.is_mxfp4:
        # Petit DequantFullScale: construct a BF16 pair, then adjust
        # its packed exponents, including zero/255 and overflow behavior.
        bits = ((packed & 0xFF) << 7) | ((packed >> 8) << 23)
        if config.gemm.high_precision:
            bits = bits + al.convert(0x03800380, al.u32)
    else:
        # NVFP4 scales are already preprocessed by a factor of 128.
        bits = al.amdgpu.perm(0, packed, 0x0C010C00) << 7
    for i in al.range(2):
        scale_bits = al.convert(bits >> (i * 16), al.u16)
        if config.gemm.is_mxfp4:
            out[0, i] = al.convert(al.bitcast(scale_bits, al.bf16), al.f32)
        else:
            out[0, i] = al.convert(al.bitcast(scale_bits, al.f16), al.f32)


@avelang.jit
def dequant(
    config: al.constexpr,
    q: al.u32,
    scale: al.f32,
    out: al.Tensor((1, 4), al.u32),
):
    if config.gemm.use_bf8:
        scale_pair = al.full((1, 2), scale, al.f32)
        bias = al.full((1, 2), config.gemm.weight_bias, al.f32)
        mask = al.convert(FP4_MASK, al.u32)
        lo = q & mask
        hi = al.amdgpu.bitreverse(q) & mask
        f0 = al.amdgpu.cvt_pk_f32_bf8(lo, 0) * bias[0]
        f1 = al.amdgpu.cvt_pk_f32_bf8(lo, 1) * bias[0]
        f2 = al.amdgpu.cvt_pk_f32_bf8(hi, 0) * bias[0]
        f3 = al.amdgpu.cvt_pk_f32_bf8(hi, 1) * bias[0]
        f0 = f0 * scale_pair[0]
        f1 = f1 * scale_pair[0]
        f2 = f2 * scale_pair[0]
        f3 = f3 * scale_pair[0]
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
    else:
        qr = al.amdgpu.bitreverse(q)
        mask = al.convert(0x8E008E00, al.u32)
        out[0, 0] = q & mask
        out[0, 1] = (q << 8) & mask
        out[0, 2] = qr & mask
        out[0, 3] = (qr << 8) & mask
        halves = al.view(out, al.f16, al.make_layout((4, 2), (2, 1)))
        if config.gemm.is_fp16:
            scale_pair = al.full((1, 2), al.convert(scale, al.f16), al.f16)
            bias = al.full((1, 2), config.gemm.weight_bias, al.f16)
            for i in al.range(4):
                halves[i] = halves[i] * bias[0]
                halves[i] = halves[i] * scale_pair[0]
        else:
            scale_pair = al.full((1, 2), scale, al.f32)
            bias = al.full((1, 2), config.gemm.weight_bias, al.f32)
            pair = al.make_local((1, 2), al.f32)
            for i in al.range(4):
                for j in al.range(2):
                    pair[0, j] = al.convert(halves[i, j], al.f32)
                pair[0] = pair[0] * bias[0]
                pair[0] = pair[0] * scale_pair[0]
                out[0, i] = al.amdgpu.perm(
                    al.bitcast(pair[0, 1], al.u32),
                    al.bitcast(pair[0, 0], al.u32), 0x07060302,
                )
