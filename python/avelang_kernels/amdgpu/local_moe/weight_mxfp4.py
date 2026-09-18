"""Native MXFP4 expert-weight descriptors and register loads.

Load helpers consume the layout produced by ExpertWeights.pack in api.py.
Their callers own waits and pipeline barriers.
"""

from functools import cache

import avelang
import avelang.language as al


@cache
def make_w13_weight_loads(config):
    """Fill [projection, half_k, n_fragment, word4] registers; caller owns waits."""
    D, I = config.compute_hidden, config.intermediate
    WN, KG, NR = config.stage1_warps_n, config.stage1_k_groups, config.stage1_wave_n // 16
    NS, CACHE = NR // 2, config.stage1_weight_load_aux

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
        group = wave // WN if KG == 2 else al.convert(0, al.u32)
        for projection in al.static_range(2):
            for half_k in al.static_range(2):
                for n in al.static_range(NR):
                    offset = (
                        (wave_n * (NR * 16) + n * 16) * (D // 2)
                        + group * (D // KG) * 8
                        + lane * 16
                        + half_k * 1024
                        + projection * I * D // 2
                    )
                    values[projection, half_k, n] = al.amdgpu.raw_buffer_load_x4(w, offset, k * 2048, CACHE)
            for n in al.static_range(NS):
                offset_s = (wave_n * NS + n) * D + group * (D // KG) + lane * 4 + projection * I * D // 32
                scales[projection, n] = al.amdgpu.raw_buffer_load_x1(ws, offset_s, k * 256, 0)

    @avelang.jit
    def prefetch_weight_phase(
        w: al.Tensor((4,), al.u32),
        ws: al.Tensor((4,), al.u32),
        values: al.Tensor((2, 2, NR, 4), al.u32),
        scales: al.Tensor((2, NS), al.u32),
        k: al.u32,
        wave: al.u32,
        lane: al.u32,
        phase: al.u32,
    ):
        wave_n = wave % WN
        group = wave // WN if KG == 2 else al.convert(0, al.u32)
        if phase == 0:
            for projection in al.static_range(2):
                offset = wave_n * D + group * (D // KG) + lane * 4 + projection * I * D // 32
                scales[projection, 0] = al.amdgpu.raw_buffer_load_x1(ws, offset, k * 256, 0)
        else:
            for load in al.static_range(8):
                if load // 3 + 1 == phase:
                    projection, n, half_k = load % 2, (load // 2) % 2, load // 4
                    offset = (
                        (wave_n * 32 + n * 16) * (D // 2)
                        + group * (D // KG) * 8
                        + lane * 16
                        + half_k * 1024
                        + projection * I * D // 2
                    )
                    values[projection, half_k, n] = al.amdgpu.raw_buffer_load_x4(w, offset, k * 2048, CACHE)

    return load_weights, prefetch_weight_phase


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
def make_w2_resources(hidden, intermediate, experts, scale_columns, *, limit_to_tile=False):
    D, I, E, SC = hidden, intermediate, experts, scale_columns

    @avelang.jit
    def initialize_w2_resources(
        weight: al.Pointer(al.u32),
        ws: al.Pointer(al.u32),
        bias_ptr: al.Pointer(al.bf16),
        expert: al.u32,
        tile: al.u32,
        bias_enabled: al.u32,
    ) -> (al.Tensor((4,), al.u32), al.Tensor((4,), al.u32), al.Tensor((4,), al.u32)):
        weights = al.make_tensor(weight, al.u32, al.make_layout((E * D * I // 8,), (1,)))
        scales = al.make_tensor(ws, al.u32, al.make_layout((E * D * SC // 4,), (1,)))
        columns = al.convert(256, al.u32) if limit_to_tile else D - tile * 256
        weight_view = al.subview(weights, ((expert * D + tile * 256) * I // 8,), (columns * I // 8,), (1,))
        scale_view = al.subview(scales, ((expert * D + tile * 256) * SC // 4,), (columns * SC // 4,), (1,))
        weight_resource = al.amdgpu.make_rsrc(weight_view, columns * I // 2)
        scale_resource = al.amdgpu.make_rsrc(scale_view, columns * SC)
        biases = al.make_tensor(bias_ptr, al.bf16, al.make_layout((E * D,), (1,)))
        bias_view = al.subview(biases, (expert * D,), (D,), (1,))
        bias_resource = al.amdgpu.make_rsrc(
            bias_view, al.select(bias_enabled != 0, al.convert(D * 2, al.u32), al.convert(0, al.u32))
        )
        return weight_resource, scale_resource, bias_resource

    return initialize_w2_resources


@cache
def make_w2_k128_weight_loads(intermediate, scale_columns, weight_cache):
    I, SC, CACHE = intermediate, scale_columns, weight_cache

    @avelang.jit
    def load_w2_values(
        resource: al.Tensor((4,), al.u32),
        values: al.Tensor((2, 4, 4), al.u32),
        k: al.u32,
        wave: al.u32,
        lane: al.u32,
    ):
        for n in al.static_range(4):
            offset = (wave * 64 + n * 16) * (I // 2) + (k % 2) * 1024 + lane * 16
            values[k % 2, n] = al.amdgpu.raw_buffer_load_x4(resource, offset, (k // 2) * 2048, CACHE)

    @avelang.jit
    def load_w2_scales(
        resource: al.Tensor((4,), al.u32),
        cached: al.Tensor((2,), al.u32),
        stages: al.Tensor((2, 2), al.u32),
        k: al.u32,
        wave: al.u32,
        lane: al.u32,
    ):
        for n in al.static_range(2):
            offset = (wave * 2 + n) * (SC * 32) + lane * 4
            cached[n] = al.amdgpu.raw_buffer_load_x1(resource, offset, (k // 2) * 256, 0)
            stages[k % 2, n] = cached[n]

    return load_w2_values, load_w2_scales
