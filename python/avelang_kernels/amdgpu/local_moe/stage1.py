"""Compute W13 with reusable M32/N32 wave register tiles."""

from functools import cache

import avelang
import avelang.language as al

from .activation import openai_swiglu, silu_dot
from .dispatch import resolve_2stage_implementation
from .input_mxfp4 import make_mxfp4_input
from .intermediate_mxfp4 import make_intermediate_store
from .solutionid import ActivationFunction
from .weight_mxfp4 import make_w13_resources, make_w13_weight_loads


@cache
def make_stage1_compute(config):
    intermediate = config.intermediate
    K_TILES = config.hidden // 256
    BM, BN, WORDS = config.stage1_tile_m, config.stage1_projection_n, config.stage1_lds_words
    MR, NR = BM // 16, BN // 64
    NS = NR // 2
    BIAS = config.bias
    SWIGLU = config.activation == ActivationFunction.OPENAI_SWIGLU
    prefetch_input, read_input = make_mxfp4_input(config)
    load_weights = make_w13_weight_loads(config)

    @avelang.jit
    def stage1_compute(
        act: al.Tensor((4,), al.u32),
        scales: al.Tensor((4,), al.u32),
        routes: al.Tensor((4,), al.u32),
        gate: al.Tensor((4,), al.u32),
        gate_scales: al.Tensor((4,), al.u32),
        up: al.Tensor((4,), al.u32),
        up_scales: al.Tensor((4,), al.u32),
        bias: al.Tensor((4,), al.u32),
        storage: al.Tensor((WORDS,), al.u32),
        tokens: al.u32,
        block: al.u32,
        wave: al.u32,
        lane: al.u32,
    ):
        accum = al.full((2, NR, MR, 4), 0, al.f32)
        fragments = al.make_local((MR, 2, 4), al.u32)
        weights = al.make_local((2, 2, NR, 4), al.u32)
        weight_scales = al.make_local((2, NS), al.u32)
        prefetch_input(act, scales, routes, storage, tokens, block, al.convert(0, al.u32), wave, lane)
        load_weights(gate, gate_scales, up, up_scales, weights, weight_scales, al.convert(0, al.u32), wave, lane)
        al.amdgpu.s_waitcnt(0, 0, 0)
        al.syncthreads()
        input_scale = read_input(storage, fragments, al.convert(0, al.u32), lane)
        for k in al.static_range(K_TILES):
            next_k = al.convert(k + 1, al.u32)
            prefetch_input(act, scales, routes, storage, tokens, block, next_k, wave, lane)
            for projection in al.static_range(2):
                for half_k in al.static_range(2):
                    for n in al.static_range(NR):
                        for m in al.static_range(MR):
                            accum[projection, n, m] = al.amdgpu.mfma_scale_16x16x128_fp4(
                                weights[projection, half_k, n],
                                weight_scales[projection, n // 2],
                                fragments[m, half_k],
                                input_scale,
                                accum[projection, n, m],
                                2 * half_k + n % 2,
                                2 * half_k + m,
                            )
            load_weights(gate, gate_scales, up, up_scales, weights, weight_scales, next_k, wave, lane)
            # Drain the next tile before reuse, including the unused terminal copy.
            al.amdgpu.s_waitcnt(0, 0, 0)
            al.syncthreads()
            if k + 1 < K_TILES:
                input_scale = read_input(storage, fragments, next_k, lane)
        result = al.view(storage, al.f32, al.make_layout((BM, BN), (BN, 1)))
        for m in al.static_range(MR):
            for n in al.static_range(NR):
                n16 = wave * NR + n
                packed_bias = al.make_local((2, 2), al.u32)
                if BIAS:
                    col = n16 // 4 * 64 + (n16 % 4) * 4 + lane // 16 * 16
                    packed_bias[0] = al.amdgpu.raw_buffer_load_x2(bias, col * 2, 0, 0)
                    packed_bias[1] = al.amdgpu.raw_buffer_load_x2(bias, (intermediate + col) * 2, 0, 0)
                bias_values = al.view(packed_bias, al.bf16, al.make_layout((2, 4), (4, 1)))
                for c in al.static_range(4):
                    gv, uv = accum[0, n, m, c], accum[1, n, m, c]
                    if BIAS:
                        gv = gv + al.convert(bias_values[0, c], al.f32)
                        uv = uv + al.convert(bias_values[1, c], al.f32)
                    value = openai_swiglu(gv, uv) if SWIGLU else silu_dot(gv, uv)
                    result[m * 16 + lane % 16, n16 * 16 + lane // 16 * 4 + c] = value
        al.syncthreads()

    return stage1_compute


def make_stage1_kernel(config):
    D, I = config.hidden, config.intermediate
    E, TOPK = config.experts, config.topk
    BM, BN, WORDS = config.stage1_tile_m, config.stage1_projection_n, config.stage1_lds_words
    SLICES, SEGMENTS = BM // 8, BN // 128
    initialize_w13_resources = make_w13_resources(config)
    stage1_compute = make_stage1_compute(config)
    store_intermediate = make_intermediate_store()

    @avelang.jit
    def stage1(
        act: al.Pointer(al.u32),
        act_scales: al.Pointer(al.u32),
        weight: al.Pointer(al.u32),
        ws: al.Pointer(al.u32),
        bias_ptr: al.Pointer(al.bf16),
        ids_ptr: al.Pointer(al.u32),
        expert_ptr: al.Pointer(al.u32),
        counts: al.Tensor((2,), al.u32),
        workspace_ptr: al.Pointer(al.u8),
        capacity: al.u32,
    ):
        extent, tokens = al.amdgpu.readfirstlane(counts[0]), al.amdgpu.readfirstlane(counts[1])
        tile, block = al.convert(al.block_id(0), al.u32), al.convert(al.block_id(1), al.u32)
        tid = al.convert(al.thread_id(0), al.u32)
        wave, lane = al.amdgpu.readfirstlane(tid // 64), tid % 64
        if block * BM >= extent:
            return
        experts = al.make_tensor(expert_ptr, al.u32, al.make_layout((capacity // BM,), (1,)))
        expert = al.amdgpu.readfirstlane(experts[block])
        if expert >= E:
            return
        routes = al.make_tensor(ids_ptr, al.u32, al.make_layout((capacity,), (1,)))
        route_view = al.subview(routes, (block * BM,), (BM,), (1,))
        route_resource = al.amdgpu.make_rsrc(route_view, BM * 4)
        input_values = al.make_tensor(act, al.u32, al.make_layout((tokens * D // 8,), (1,)))
        input_scales = al.make_tensor(act_scales, al.u32, al.make_layout((capacity * D // 128,), (1,)))
        act_resource = al.amdgpu.make_rsrc(input_values, tokens * D // 2)
        scale_resource = al.amdgpu.make_rsrc(input_scales, capacity * D // 32)
        wg, sg, bias = initialize_w13_resources(weight, ws, bias_ptr, expert, tile, D, I, al.convert(0, al.u32))
        wu, su, _ = initialize_w13_resources(weight, ws, bias_ptr, expert, tile, D, I, al.convert(1, al.u32))
        storage = al.make_shared((WORDS,), al.u32)
        stage1_compute(
            act_resource, scale_resource, route_resource, wg, sg, wu, su, bias, storage, tokens, block, wave, lane
        )
        scale_base = capacity * I // 2
        workspace_bytes = scale_base + ((capacity + 255) // 256) * 256 * (I // 32)
        memory = al.make_tensor(workspace_ptr, al.u8, al.make_layout((workspace_bytes,), (1,)))
        output_resource = al.amdgpu.make_rsrc(memory, workspace_bytes)
        values = al.view(storage, al.f32, al.make_layout((BM, BN // 4, 4), (BN, 4, 1)))
        for batch in al.static_range(SLICES):
            row = batch * 8 + tid // 32
            route = al.amdgpu.raw_buffer_load_x1(route_resource, row * 4, 0, 0)
            token, slot = route & 0xFFFFFF, route >> 24
            if block * BM + row < extent and token < tokens and slot < TOPK:
                for segment in al.static_range(SEGMENTS):
                    col_lane = segment * 32 + tid % 32
                    fragment = values[row, col_lane]
                    store_intermediate(
                        fragment,
                        output_resource,
                        token * TOPK + slot,
                        block * BM + row,
                        tile * BN,
                        col_lane,
                        scale_base,
                        I,
                    )

    return stage1


@cache
def make_stage1(config):
    return resolve_2stage_implementation(config)[0](config)
