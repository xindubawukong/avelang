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
def make_w13_resources(config):
    E, BN = config.experts, config.stage1_projection_n

    @avelang.jit
    def initialize_w13_resources(
        weight: al.Pointer(al.u32),
        scales: al.Pointer(al.u32),
        bias: al.Pointer(al.bf16),
        expert: al.u32,
        tile: al.u32,
        hidden: al.u32,
        intermediate: al.u32,
        projection: al.u32,
    ) -> (al.Tensor((4,), al.u32), al.Tensor((4,), al.u32), al.Tensor((4,), al.u32)):
        values = al.make_tensor(weight, al.u32, al.make_layout((E * intermediate * hidden // 4,), (1,)))
        scale_values = al.make_tensor(scales, al.u32, al.make_layout((E * intermediate * hidden // 64,), (1,)))
        base_n = (expert * 2 + projection) * intermediate + tile * BN
        w_view = al.subview(values, (base_n * hidden // 8,), (BN * hidden // 8,), (1,))
        s_view = al.subview(scale_values, (base_n * hidden // 128,), (BN * hidden // 128,), (1,))
        biases = al.make_tensor(bias, al.bf16, al.make_layout((E * 2 * intermediate,), (1,)))
        b_view = al.subview(biases, (expert * 2 * intermediate + tile * BN,), (2 * intermediate - tile * BN,), (1,))
        return (
            al.amdgpu.make_rsrc(w_view, BN * hidden // 2),
            al.amdgpu.make_rsrc(s_view, BN * hidden // 32),
            al.amdgpu.make_rsrc(b_view, (2 * intermediate - tile * BN) * 2),
        )

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
    D, NR = config.hidden, config.stage1_projection_n // 64
    NS = NR // 2

    @avelang.jit
    def load_weights(
        gate: al.Tensor((4,), al.u32),
        gate_scales: al.Tensor((4,), al.u32),
        up: al.Tensor((4,), al.u32),
        up_scales: al.Tensor((4,), al.u32),
        weights: al.Tensor((2, 2, NR, 4), al.u32),
        scales: al.Tensor((2, NS), al.u32),
        k: al.u32,
        wave: al.u32,
        lane: al.u32,
    ):
        for half_k in al.static_range(2):
            for n in al.static_range(NR):
                n16 = wave * NR + n
                weights[0, half_k, n] = load_weight_values(gate, D, n16, k * 2 + half_k, lane)
                weights[1, half_k, n] = load_weight_values(up, D, n16, k * 2 + half_k, lane)
        for n in al.static_range(NS):
            scales[0, n] = load_weight_scales(gate_scales, D, wave * NS + n, k, lane)
            scales[1, n] = load_weight_scales(up_scales, D, wave * NS + n, k, lane)

    return load_weights
