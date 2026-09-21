"""Native FP4 values and blocked scales with separate W13 descriptors."""

from functools import cache

import avelang
import avelang.language as al


@avelang.jit
def load_weight_values(
    weight: al.Tensor((4,), al.u32),
    columns: al.u32,
    n16: al.u32,
    k128: al.u32,
    lane: al.u32,
) -> al.Tensor((4,), al.u32):
    offset = n16 * 16 * (columns // 2) + k128 * 1024 + lane * 16
    return al.amdgpu.raw_buffer_load_x4(weight, offset, 0, 0)


@avelang.jit
def load_weight_scales(
    scales: al.Tensor((4,), al.u32),
    columns: al.u32,
    n32: al.u32,
    k256: al.u32,
    lane: al.u32,
) -> al.u32:
    offset = n32 * (columns // 256) * 256 + k256 * 256 + lane * 4
    return al.amdgpu.raw_buffer_load_x1(scales, offset, 0, 0)


@cache
def make_w13_resources(hidden, intermediate, experts, projection_n, bias_stride):
    D, I, E, BN, BIAS_STRIDE = hidden, intermediate, experts, projection_n, bias_stride

    @avelang.jit
    def initialize_w13_resources(
        weight: al.Pointer(al.u32),
        ws: al.Pointer(al.u32),
        bias_ptr: al.Pointer(al.bf16),
        expert: al.u32,
        tile: al.u32,
        bias_enabled: al.u32,
    ) -> (al.Tensor((4,), al.u32), al.Tensor((4,), al.u32), al.Tensor((4,), al.u32)):
        weights = al.make_tensor(weight, al.u32, al.make_layout((E * (I * D // 4),), (1,)))
        scales = al.make_tensor(ws, al.u32, al.make_layout((E * (I * D // 64),), (1,)))
        weight_view = al.subview(weights, (expert * (I * D // 4) + tile * BN * D // 8,), ((I + BN) * D // 8,), (1,))
        scale_view = al.subview(scales, (expert * (I * D // 64) + tile * BN * D // 128,), ((I + BN) * D // 128,), (1,))
        weight_resource = al.amdgpu.make_rsrc(weight_view, (I + BN) * D // 2)
        scale_resource = al.amdgpu.make_rsrc(scale_view, (I + BN) * D // 32)
        biases = al.make_tensor(bias_ptr, al.bf16, al.make_layout((E * 2 * BIAS_STRIDE,), (1,)))
        bias_view = al.subview(biases, (expert * 2 * BIAS_STRIDE + tile * BN,), (BN,), (1,))
        bias_resource = al.amdgpu.make_rsrc(
            bias_view, al.select(bias_enabled != 0, (2 * BIAS_STRIDE - tile * BN) * 2, al.convert(0, al.u32))
        )
        return weight_resource, scale_resource, bias_resource

    return initialize_w13_resources


@cache
def make_w2_resources(config):
    E = config.experts

    @avelang.jit
    def initialize_w2_resources(
        weight: al.Pointer(al.u32),
        scales: al.Pointer(al.u32),
        bias: al.Pointer(al.bf16),
        expert: al.u32,
        tile: al.u32,
        hidden: al.u32,
        intermediate: al.u32,
    ) -> (al.Tensor((4,), al.u32), al.Tensor((4,), al.u32), al.Tensor((4,), al.u32)):
        values = al.make_tensor(weight, al.u32, al.make_layout((E * hidden * intermediate // 8,), (1,)))
        scale_values = al.make_tensor(scales, al.u32, al.make_layout((E * hidden * intermediate // 128,), (1,)))
        w_view = al.subview(
            values, ((expert * hidden + tile * 256) * intermediate // 8,), (256 * intermediate // 8,), (1,)
        )
        s_view = al.subview(
            scale_values, ((expert * hidden + tile * 256) * intermediate // 128,), (256 * intermediate // 128,), (1,)
        )
        biases = al.make_tensor(bias, al.bf16, al.make_layout((E * hidden,), (1,)))
        b_view = al.subview(biases, (expert * hidden,), (hidden,), (1,))
        return (
            al.amdgpu.make_rsrc(w_view, 256 * intermediate // 2),
            al.amdgpu.make_rsrc(s_view, 256 * intermediate // 32),
            al.amdgpu.make_rsrc(b_view, hidden * 2),
        )

    return initialize_w2_resources


@cache
def make_w13_weight_loads(config):
    """Fill [projection, half_k, n_fragment, word4] registers; caller owns waits."""
    D, I = config.compute_hidden, config.intermediate
    WN, NR = config.stage1_warps_n, config.stage1_wave_n // 16
    NS, CACHE = NR // 2, config.weight_load_aux

    @avelang.jit
    def load_weights(
        w: al.Tensor((4,), al.u32),
        ws: al.Tensor((4,), al.u32),
        values: al.Tensor((2, 2, NR, 4), al.u32),
        scales: al.Tensor((2, NS), al.u32),
        k: al.u32,
        wave: al.u32,
        lane: al.u32,
    ):
        wave_n = wave % WN
        for projection in al.static_range(2):
            for half_k in al.static_range(2):
                for n in al.static_range(NR):
                    offset = (
                        (wave_n * (NR * 16) + n * 16) * (D // 2) + lane * 16 + half_k * 1024 + projection * I * D // 2
                    )
                    values[projection, half_k, n] = al.amdgpu.raw_buffer_load_x4(w, offset + k * 2048, 0, CACHE)
            for n in al.static_range(NS):
                offset_s = (wave_n * NS + n) * D + lane * 4 + projection * I * D // 32
                scales[projection, n] = al.amdgpu.raw_buffer_load_x1(ws, offset_s + k * 256, 0, 0)

    return load_weights
