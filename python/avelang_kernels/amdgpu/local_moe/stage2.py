"""Compute W2 with reusable M32/N64 wave register tiles."""

from functools import cache

import avelang
import avelang.language as al

from .dispatch import resolve_2stage_implementation
from .intermediate_mxfp4 import make_stage2_input
from .weight_mxfp4 import load_weight_fragment, make_w2_resources


@cache
def make_stage2_compute(config):
    intermediate = config.intermediate
    K_TILES = config.intermediate // 128
    BIAS = config.bias
    load_intermediate = make_stage2_input(config)

    @avelang.jit
    def stage2_compute(
        act: al.Tensor((4,), al.u32),
        routes: al.Tensor((4,), al.u32),
        weight: al.Tensor((4,), al.u32),
        scales: al.Tensor((4,), al.u32),
        bias: al.Tensor((4,), al.u32),
        route_weights: al.Tensor((4,), al.u32),
        storage: al.Tensor((4096,), al.u32),
        tokens: al.u32,
        block: al.u32,
        scale_base: al.u32,
        tile: al.u32,
        wave: al.u32,
        lane: al.u32,
    ):
        accum = al.full((4, 2, 4), 0, al.f32)
        fragments = al.make_local((2, 4), al.u32)
        input_scales = al.make_local((2,), al.u32)
        weights = al.make_local((4, 4), al.u32)
        weight_scales = al.make_local((4,), al.u32)
        for k in al.static_range(K_TILES):
            ki = al.convert(k, al.u32)
            for m in al.static_range(2):
                xv, xs = load_intermediate(
                    act, routes, intermediate, tokens, block, al.convert(m, al.u32), ki, lane, scale_base
                )
                fragments[m], input_scales[m] = xv, xs
            for n in al.static_range(4):
                n16 = wave * 4 + n
                wv, ws = load_weight_fragment(weight, scales, intermediate, n16, ki, lane)
                weights[n], weight_scales[n] = wv, ws
            al.amdgpu.s_waitcnt(0, 0, 0)
            for n in al.static_range(4):
                for m in al.static_range(2):
                    accum[n, m] = al.amdgpu.mfma_scale_16x16x128_fp4(
                        weights[n], weight_scales[n], fragments[m], input_scales[m], accum[n, m], 0, 0
                    )
        result = al.view(storage, al.bf16, al.make_layout((32, 256), (256, 1)))
        for m in al.static_range(2):
            for n in al.static_range(4):
                n16 = wave * 4 + n
                packed_bias = al.make_local((2,), al.u32)
                if BIAS:
                    col = tile * 256 + n16 // 4 * 64 + n16 % 4 * 4 + lane // 16 * 16
                    packed_bias = al.amdgpu.raw_buffer_load_x2(bias, col * 2, 0, 0)
                bias_values = al.view(packed_bias, al.bf16, al.make_layout((4,), (1,)))
                bits = al.amdgpu.raw_buffer_load_x1(route_weights, (m * 16 + lane % 16) * 4, 0, 0)
                rw = al.bitcast(bits, al.f32)
                for c in al.static_range(4):
                    value = accum[n, m, c]
                    if BIAS:
                        value = value + al.convert(bias_values[c], al.f32)
                    result[m * 16 + lane % 16, n16 * 16 + lane // 16 * 4 + c] = al.convert(value * rw, al.bf16)
        al.syncthreads()

    return stage2_compute


def make_stage2_kernel(config):
    D, I = config.hidden, config.intermediate
    TOPK = config.topk
    initialize_w2_resources = make_w2_resources(config)
    stage2_compute = make_stage2_compute(config)

    @avelang.jit
    def stage2(
        workspace_ptr: al.Pointer(al.u8),
        weight: al.Pointer(al.u32),
        ws: al.Pointer(al.u32),
        bias_ptr: al.Pointer(al.bf16),
        ids_ptr: al.Pointer(al.u32),
        expert_ptr: al.Pointer(al.u32),
        route_weight_ptr: al.Pointer(al.f32),
        counts: al.Tensor((2,), al.u32),
        out_ptr: al.Pointer(al.bf16),
        capacity: al.u32,
    ):
        tile, expert = al.convert(al.block_id(0), al.u32), al.convert(al.block_id(1), al.u32)
        tid = al.convert(al.thread_id(0), al.u32)
        wave, lane = al.amdgpu.readfirstlane(tid // 64), tid % 64
        extent, tokens = al.amdgpu.readfirstlane(counts[0]), al.amdgpu.readfirstlane(counts[1])
        storage = al.make_shared((4096,), al.u32)
        experts = al.make_tensor(expert_ptr, al.u32, al.make_layout((capacity // 32,), (1,)))
        for block in al.range((extent + 31) // 32):
            group = al.convert(block, al.u32)
            if al.amdgpu.readfirstlane(experts[group]) == expert:
                routes = al.make_tensor(ids_ptr, al.u32, al.make_layout((capacity,), (1,)))
                route_weights = al.make_tensor(route_weight_ptr, al.f32, al.make_layout((capacity,), (1,)))
                route_view = al.subview(routes, (group * 32,), (32,), (1,))
                rw_view = al.subview(route_weights, (group * 32,), (32,), (1,))
                route_resource = al.amdgpu.make_rsrc(route_view, 128)
                rw_resource = al.amdgpu.make_rsrc(rw_view, 128)
                scale_base = capacity * I // 2
                workspace_bytes = scale_base + ((capacity + 255) // 256) * 256 * (I // 32)
                memory = al.make_tensor(workspace_ptr, al.u8, al.make_layout((workspace_bytes,), (1,)))
                act_resource = al.amdgpu.make_rsrc(memory, workspace_bytes)
                wr, sr, br = initialize_w2_resources(weight, ws, bias_ptr, expert, tile, D, I)
                stage2_compute(
                    act_resource,
                    route_resource,
                    wr,
                    sr,
                    br,
                    rw_resource,
                    storage,
                    tokens,
                    group,
                    scale_base,
                    tile,
                    wave,
                    lane,
                )
                output = al.make_tensor(out_ptr, al.bf16, al.make_layout((tokens * D,), (1,)))
                output_resource = al.amdgpu.make_rsrc(output, tokens * D * 2)
                pairs = al.view(storage, al.bf16, al.make_layout((4096, 2), (2, 1)))
                for m in al.static_range(16):
                    pair = m * 256 + tid
                    row, column = pair // 128, pair % 128 * 2
                    route = al.amdgpu.raw_buffer_load_x1(route_resource, row * 4, 0, 0)
                    token, slot = route & 0xFFFFFF, route >> 24
                    if group * 32 + row < extent and token < tokens and slot < TOPK:
                        offset = (token * D + tile * 256 + column) * 2
                        al.amdgpu.raw_buffer_atomic_add_bf16x2(pairs[pair], output_resource, offset)
                al.syncthreads()

    return stage2


@cache
def make_stage2(config):
    return resolve_2stage_implementation(config)[1](config)
