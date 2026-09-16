"""Device-side FP4 dequantization helpers for AMDGPU."""

import avelang
import avelang.language as al

from .config import FP4GemmConfig
from .solution import MatmulElementB, MatmulMfmaType

FP4_MASK = 0x8E8E8E8E


def make_fp4_dequant(config: FP4GemmConfig):
    IS_MXFP4 = config.element_b == MatmulElementB.MXFP4
    IS_FP16 = config.mfma_type == MatmulMfmaType.FP16
    HIGH_PRECISION = config.high_precision
    USE_BF8 = config.use_bf8
    WEIGHT_BIAS = config.dequant_bias / (1.0 if IS_MXFP4 else 128.0)
    BF8_MULTIPLIER = WEIGHT_BIAS if HIGH_PRECISION else 1.0

    @avelang.jit
    def _dequant_scales(
        packed_scale: al.u16,
        out: al.Tensor((1, 2), al.f32),
    ):
        for i in al.range(2):
            scale_byte = (al.convert(packed_scale, al.u32) >> (i * 8)) & 0xFF
            if IS_MXFP4:
                # E8M0: 0 encodes 2**-127, 255 encodes NaN.
                bits = scale_byte << 23
                bits = al.select(scale_byte == 0, 0x00400000, bits)
                bits = al.select(scale_byte == 255, 0x7FC00000, bits)
                out[0, i] = al.bitcast(bits, al.f32)
            else:
                scale_bits = al.convert(scale_byte << 7, al.u16)
                out[0, i] = al.convert(al.bitcast(scale_bits, al.f16), al.f32)

    @avelang.jit
    def _dequant(
        q: al.u32,
        scale: al.f32,
        out: al.Tensor((1, 4), al.u32),
    ):
        if USE_BF8:
            scale_pair = al.full((1, 2), scale, al.f32)
            bias = al.full((1, 2), BF8_MULTIPLIER, al.f32)
            mask = al.convert(FP4_MASK, al.u32)
            lo = q & mask
            hi = al.amdgpu.bitreverse(q) & mask
            # Restore weights before multiplying MX scales to avoid
            # overflowing large E8M0 scales with an intermediate upscale.
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
            if IS_FP16:
                scale_pair = al.full((1, 2), al.convert(scale, al.f16), al.f16)
                for i in al.range(4):
                    if HIGH_PRECISION:
                        bias = al.full((1, 2), WEIGHT_BIAS, al.f16)
                        halves[i] = halves[i] * bias[0]
                    halves[i] = halves[i] * scale_pair[0]
            else:
                scale_pair = al.full((1, 2), scale, al.f32)
                pair = al.make_local((1, 2), al.f32)
                for i in al.range(4):
                    for j in al.range(2):
                        pair[0, j] = al.convert(halves[i, j], al.f32)
                    if HIGH_PRECISION:
                        bias = al.full((1, 2), WEIGHT_BIAS, al.f32)
                        pair[0] = pair[0] * bias[0]
                    pair[0] = pair[0] * scale_pair[0]
                    out[0, i] = al.amdgpu.perm(
                        al.bitcast(pair[0, 1], al.u32),
                        al.bitcast(pair[0, 0], al.u32), 0x07060302,
                    )

    return _dequant_scales, _dequant
