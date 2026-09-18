"""Native FP4 values and blocked scales with separate W13 descriptors."""

from functools import cache

import avelang
import avelang.language as al


@avelang.jit
def load_weight_fragment(
    weight: al.Tensor((4,), al.u32),
    scales: al.Tensor((4,), al.u32),
    columns: al.u32,
    n16: al.u32,
    k128: al.u32,
    lane: al.u32,
) -> (al.Tensor((4,), al.u32), al.u32):
    offset = n16 * 16 * (columns // 2) + k128 * 1024 + lane * 16
    values = al.amdgpu.raw_buffer_load_x4(weight, offset, 0, 0)
    offset_s = n16 // 2 * (columns // 256) * 256 + k128 // 2 * 256 + lane * 4
    word = al.amdgpu.raw_buffer_load_x1(scales, offset_s, 0, 0)
    scale = (word >> ((2 * (k128 % 2) + n16 % 2) * 8)) & 255
    return values, scale


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
