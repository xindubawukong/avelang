"""Compute W13 one MFMA cell at a time, then quantize its logical output tile."""

from functools import cache

import avelang
import avelang.language as al

from .activation import openai_swiglu, silu_dot
from .dispatch import resolve_2stage_implementation
from .input_mxfp4 import make_mxfp4_input
from .intermediate_mxfp4 import make_intermediate_store
from .solutionid import ActivationFunction
from .weight_mxfp4 import load_weight_fragment, make_w13_resources


@cache
def make_stage1_compute(config):
    BM, BN, WORDS = config.stage1_tile_m, config.stage1_projection_n, config.stage1_lds_words
    M_CELLS, N_PASSES = BM // 16, BN // 64
    BIAS = config.bias
    SWIGLU = config.activation == ActivationFunction.OPENAI_SWIGLU
    load_input_fragment = make_mxfp4_input(config)

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
        hidden: al.u32,
        intermediate: al.u32,
        tokens: al.u32,
        block: al.u32,
        wave: al.u32,
        lane: al.u32,
    ):
        result = al.view(storage, al.f32, al.make_layout((BM, BN), (BN, 1)))
        for m in al.static_range(M_CELLS):
            for n in al.static_range(N_PASSES):
                n16 = wave + n * 4
                g = al.full((4,), 0, al.f32)
                u = al.full((4,), 0, al.f32)
                for k in al.range(hidden // 128):
                    ki = al.convert(k, al.u32)
                    xv, xs = load_input_fragment(
                        act, scales, routes, hidden, tokens, block, al.convert(m, al.u32), ki, lane
                    )
                    wg, sg = load_weight_fragment(gate, gate_scales, hidden, n16, ki, lane)
                    wu, su = load_weight_fragment(up, up_scales, hidden, n16, ki, lane)
                    al.amdgpu.s_waitcnt(0, 0, 0)
                    g = al.amdgpu.mfma_scale_16x16x128_fp4(wg, sg, xv, xs, g, 0, 0)
                    u = al.amdgpu.mfma_scale_16x16x128_fp4(wu, su, xv, xs, u, 0, 0)
                packed_bias = al.make_local((2, 2), al.u32)
                if BIAS:
                    col = n16 // 4 * 64 + (n16 % 4) * 4 + lane // 16 * 16
                    packed_bias[0] = al.amdgpu.raw_buffer_load_x2(bias, col * 2, 0, 0)
                    packed_bias[1] = al.amdgpu.raw_buffer_load_x2(bias, (intermediate + col) * 2, 0, 0)
                bias_values = al.view(packed_bias, al.bf16, al.make_layout((2, 4), (4, 1)))
                for c in al.static_range(4):
                    gv, uv = g[c], u[c]
                    if BIAS:
                        gv = gv + al.convert(bias_values[0, c], al.f32)
                        uv = uv + al.convert(bias_values[1, c], al.f32)
                    value = openai_swiglu(gv, uv) if SWIGLU else silu_dot(gv, uv)
                    result[m * 16 + lane % 16, n16 * 16 + lane // 16 * 4 + c] = value
        al.syncthreads()

    return stage1_compute


def make_stage1_kernel(config):
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
        hidden_dim: al.u32,
        intermediate_dim: al.u32,
    ):
        D, I = hidden_dim, intermediate_dim
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
            act_resource, scale_resource, route_resource, wg, sg, wu, su, bias, storage, D, I, tokens, block, wave, lane
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
