"""Row-packed MXFP4 transport quantizer: one thread per 32 BF16 values."""

from functools import cache

import avelang
import avelang.language as al
import torch


@avelang.jit
def _round_up_e8m0(maximum: al.f32) -> al.u32:
    """E8M0 exponent for the transport quantizer's upward-rounded amax / 6."""
    bits = al.bitcast(maximum * al.convert(1.0 / 6.0, al.f32), al.u32)
    exponent = (bits >> 23) & 255
    increment = al.select((exponent < 255) and ((bits & 0x7FFFFF) != 0), al.convert(1, al.u32), al.convert(0, al.u32))
    return exponent + increment


@cache
def _make_mxfp4_quantizer(columns):
    GROUPS = columns // 32
    SCALES = (GROUPS + 15) // 16 * 16
    STRIDE = columns // 2 + SCALES

    @avelang.jit
    def quantize(x: al.Pointer(al.bf16), q: al.Pointer(al.u8), rows: al.u32, input_stride: al.u32):
        row = al.convert(al.block_id(1), al.u32)
        group = al.convert(al.block_id(0) * 64 + al.thread_id(0), al.u32)
        output = al.make_tensor(q, al.u8, al.make_layout((rows, STRIDE), (STRIDE, 1)))
        if group < SCALES:
            exponent = al.convert(0, al.u32)
            if group < GROUPS:
                source = al.make_tensor(x, al.bf16, al.make_layout((rows, GROUPS, 4, 8), (input_stride, 32, 8, 1)))
                values = al.make_local((8, 4), al.f32)
                maximum = al.convert(0.0, al.f32)
                for chunk in al.static_range(4):
                    packed = source[row, group, chunk]
                    for c in al.static_range(8):
                        values[chunk * 2 + c // 4, c % 4] = al.convert(packed[c], al.f32)
                for v in al.static_range(8):
                    m = al.max(
                        al.max(al.abs(values[v, 0]), al.abs(values[v, 1])),
                        al.max(al.abs(values[v, 2]), al.abs(values[v, 3])),
                    )
                    maximum = al.max(maximum, m)
                exponent = _round_up_e8m0(maximum)
                scale = al.bitcast(exponent << 23, al.f32)
                result = al.make_local((4,), al.u32)
                for word in al.static_range(4):
                    value = al.convert(0, al.u32)
                    for pair in al.static_range(4):
                        vector, c = word * 2 + pair // 2, (pair % 2) * 2
                        value = al.amdgpu.cvt_scalef32_pk_fp4_f32(
                            value, values[vector, c], values[vector, c + 1], scale, pair
                        )
                    result[word] = al.select(maximum == 0.0, al.convert(0, al.u32), value)
                output32 = al.view(output, al.u32, al.make_layout((rows, STRIDE // 16, 4), (STRIDE // 4, 4, 1)))
                output32[row, group] = result
            output[row, columns // 2 + group] = al.convert(exponent, al.u8)

    return quantize


def quantize_input(x, *, out=None):
    if x.ndim != 2 or x.dtype != torch.bfloat16 or x.device.type != "cuda":
        raise ValueError("input must be a GPU BF16 matrix")
    rows, columns = x.shape
    if columns <= 0 or columns % 64 or x.stride(1) != 1 or x.stride(0) % 8:
        raise ValueError("input columns must be a positive multiple of 64 with contiguous, 16-byte aligned rows")
    stride = columns // 2 + (columns // 32 + 15) // 16 * 16
    if out is None:
        storage = torch.empty((rows, stride), dtype=torch.uint8, device=x.device)
        q, scales = storage[:, : columns // 2], storage[:, columns // 2 : columns // 2 + columns // 32]
    else:
        q, scales = out
        for value, shape in ((q, (rows, columns // 2)), (scales, (rows, columns // 32))):
            if (
                value.shape != shape
                or value.dtype != torch.uint8
                or value.device != x.device
                or value.stride() != (stride, 1)
            ):
                raise ValueError(
                    "quantization outputs must be row-packed GPU uint8 views with matching shape and strides"
                )
        if rows and scales.data_ptr() != q.data_ptr() + columns // 2:
            raise ValueError("scales must be a view into the output rows")
    if rows:
        _make_mxfp4_quantizer(columns)[lambda: (((columns // 32 + 63) // 64, rows, 1), (64, 1, 1))](
            x, q, rows, x.stride(0)
        )
    return q, scales
