"""Stage1 compute shared with MegaMoE, followed by the local kernel wrapper.

The compute pipeline produces an FP32 activation tile in LDS. Input-format
callbacks supply act/scale fragments; the wrapper owns routing, validity,
intermediate stores and any readiness publication.
"""

from functools import cache

import avelang
import avelang.language as al

from .activation import openai_swiglu, silu_dot, situ_v2
from .config import MoeConfig
from .dispatch import resolve_2stage_implementation
from .input_mxfp4 import make_mxfp4_input
from .intermediate_mxfp4 import make_intermediate_store
from .solutionid import ActivationFunction
from .weight_mxfp4 import make_w13_resources, make_w13_weight_loads


@cache
def make_stage1_compute(config, *, prefetch_input, read_input):
    """Write FP32 [tile_m, projection_n] activations into the shared arena.

    The input callbacks issue loads and read fragments; this pipeline owns
    their waits. The caller must synchronize before consuming the final tile.
    """
    D, I = config.compute_hidden, config.intermediate
    BM, BN, WN, WM = config.stage1_tile_m, config.stage1_projection_n, config.stage1_warps_n, config.stage1_wave_m
    KG, NR, MR = config.stage1_k_groups, config.stage1_wave_n // 16, WM // 16
    NS = NR // 2
    WORDS = config.stage1_lds_words
    BIAS = config.bias
    SWIGLU = config.activation == ActivationFunction.OPENAI_SWIGLU
    SITU = config.activation == ActivationFunction.SITU_V2
    K_TILES = D // KG // 256
    BIAS_STRIDE = (I + 255) // 256 * 256
    load_weights = make_w13_weight_loads(config)

    @avelang.jit
    def stage1_compute(
        act_resource: al.Tensor((4,), al.u32),
        act_scale_resource: al.Tensor((4,), al.u32),
        resource_w: al.Tensor((4,), al.u32),
        resource_ws: al.Tensor((4,), al.u32),
        bias_resource: al.Tensor((4,), al.u32),
        storage: al.Tensor((WORDS,), al.u32),
        input_offsets: al.Tensor((2,), al.u32),
        block: al.u32,
        wave: al.u32,
        lane: al.u32,
        tid: al.u32,
    ):
        wave_n = wave % WN
        wave_m = al.convert(0, al.u32) if KG == 2 else wave // WN
        accum = al.make_local((2, NR, MR, 4), al.f32)
        for projection in al.static_range(2):
            for n in al.static_range(NR):
                for m in al.static_range(MR):
                    for c in al.static_range(4):
                        accum[projection, n, m, c] = al.convert(0.0, al.f32)
        weights = al.make_local((2, 2, NR, 4), al.u32)
        weight_scales = al.make_local((2, NS), al.u32)
        fragments = al.make_local((MR, 2, 4), al.u32)
        input_scales = al.make_local((2,), al.u32)
        prefetch_input(
            act_resource, act_scale_resource, storage, input_offsets, block, al.convert(0, al.u32), wave, lane
        )
        load_weights(resource_w, resource_ws, weights, weight_scales, al.convert(0, al.u32), wave, lane)
        al.amdgpu.s_waitcnt(0, 0, 0)
        al.syncthreads()
        read_input(storage, fragments, input_scales, al.convert(0, al.u32), wave, lane)
        for k in al.static_range(K_TILES):
            if k + 1 < K_TILES:
                prefetch_input(
                    act_resource,
                    act_scale_resource,
                    storage,
                    input_offsets,
                    block,
                    al.convert(k + 1, al.u32),
                    wave,
                    lane,
                )
            for projection in al.static_range(2):
                for half_k in al.static_range(2):
                    for n in al.static_range(NR):
                        for m in al.static_range(MR):
                            accum[projection, n, m] = al.amdgpu.mfma_scale_16x16x128_fp4(
                                weights[projection, half_k, n],
                                weight_scales[projection, n // 2],
                                fragments[m, half_k],
                                input_scales[m // 2],
                                accum[projection, n, m],
                                2 * half_k + n % 2,
                                2 * half_k + m % 2,
                            )
            if k + 1 < K_TILES:
                load_weights(resource_w, resource_ws, weights, weight_scales, al.convert(k + 1, al.u32), wave, lane)
                al.amdgpu.s_waitcnt(0, 0, 0)
                al.syncthreads()
                read_input(storage, fragments, input_scales, al.convert(k + 1, al.u32), wave, lane)
        if KG == 2:
            partials = al.view(
                storage, al.f32, al.make_layout((2, NR, MR, 128, 4), (NR * MR * 512, MR * 512, 512, 4, 1))
            )
            al.syncthreads()
            if wave >= 2:
                for projection in al.static_range(2):
                    for n in al.static_range(NR):
                        for m in al.static_range(MR):
                            partials[projection, n, m, tid - 128] = accum[projection, n, m]
            al.syncthreads()
            if wave < 2:
                for projection in al.static_range(2):
                    for n in al.static_range(NR):
                        for m in al.static_range(MR):
                            accum[projection, n, m] = accum[projection, n, m] + partials[projection, n, m, tid]
        if BIAS:
            for projection in al.static_range(2):
                packed_bias = al.make_local((NR, 2), al.u32)
                for n in al.static_range(NR):
                    col = wave_n * (NR * 16) + n * 16
                    offset_b = al.convert(
                        (projection * BIAS_STRIDE + (col // 64) * 64 + ((col % 64) // 16) * 4 + (lane // 16) * 16) * 2,
                        al.u32,
                    )
                    packed_bias[n] = al.amdgpu.raw_buffer_load_x2(bias_resource, offset_b, 0, 0)
                bias_values = al.view(packed_bias, al.bf16, al.make_layout((NR, 4), (4, 1)))
                for n in al.static_range(NR):
                    for m in al.static_range(MR):
                        for c in al.static_range(4):
                            accum[projection, n, m, c] = accum[projection, n, m, c] + al.convert(
                                bias_values[n, c], al.f32
                            )
        hidden = al.view(storage, al.f32, al.make_layout((BM, BN // 4, 4), (BN, 4, 1)))
        activated = al.make_local((NR, MR, 4), al.f32)
        for n in al.range(NR):
            for m in al.range(MR):
                for c in al.range(4):
                    gate, up = accum[0, n, m, c], accum[1, n, m, c]
                    if SITU:
                        activated[n, m, c] = situ_v2(gate, up)
                    elif SWIGLU:
                        activated[n, m, c] = openai_swiglu(gate, up)
                    else:
                        activated[n, m, c] = silu_dot(gate, up)
        al.syncthreads()
        if KG == 1 or wave < 2:
            for n in al.static_range(NR):
                for m in al.static_range(MR):
                    hidden[wave_m * WM + m * 16 + lane % 16, wave_n * (NR * 4) + n * 4 + lane // 16] = activated[n, m]

    return stage1_compute


def make_stage1_kernel(config: MoeConfig):
    """Build the local Stage1 kernel with routing and intermediate stores."""
    D, I, E, TOPK = config.hidden, config.intermediate, config.experts, config.topk
    BM, BN, WM = config.stage1_tile_m, config.stage1_projection_n, config.stage1_wave_m
    TB = BM // 4
    WORDS, ARENA = config.stage1_lds_words, config.stage1_arena_words
    SORTED = config.sorted_intermediate
    SCALE_COLS, BIAS_STRIDE = config.scale_columns, (I + 255) // 256 * 256
    COL_LANES = min(BN // 4, 32)
    ROWS_PER_SLICE = 256 // COL_LANES
    SLICES, SEGMENTS = BM // ROWS_PER_SLICE, (BN + 127) // 128
    prefetch_input, read_input = make_mxfp4_input(config)
    stage1_compute = make_stage1_compute(config, prefetch_input=prefetch_input, read_input=read_input)
    initialize_w13_resources = make_w13_resources(D, I, E, BN, BIAS_STRIDE)
    INPUT_LOADS = BM // 32
    store_intermediate = make_intermediate_store(I, SCALE_COLS, act_aux=2, scale_aux=0)

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
        route_extent = al.amdgpu.readfirstlane(counts[0])
        num_tokens = al.amdgpu.readfirstlane(counts[1])
        tile, block = al.convert(al.block_id(0), al.u32), al.convert(al.block_id(1), al.u32)
        tid = al.convert(al.thread_id(0), al.u32)
        lane = tid % 64
        wave = al.amdgpu.readfirstlane(tid // 64)
        if block * BM >= route_extent:
            return
        experts = al.make_tensor(expert_ptr, al.u32, al.make_layout((capacity // BM,), (1,)))
        expert = al.amdgpu.readfirstlane(experts[block])
        if expert >= E:
            return
        routes = al.make_tensor(ids_ptr, al.u32, al.make_layout((capacity,), (1,)))
        ACT = al.make_tensor(act, al.u32, al.make_layout((num_tokens * D // 8,), (1,)))
        ACT_SCALES = al.make_tensor(act_scales, al.u32, al.make_layout((capacity * D // 128,), (1,)))
        resource_w, resource_ws, bias_resource = initialize_w13_resources(
            weight, ws, bias_ptr, expert, tile, al.convert(1, al.u32)
        )
        scale_base = capacity * I // 2
        workspace_bytes = scale_base + ((capacity + 255) // 256) * 256 * SCALE_COLS
        workspace = al.make_tensor(workspace_ptr, al.u8, al.make_layout((workspace_bytes,), (1,)))
        output_resource = al.amdgpu.make_rsrc(workspace, workspace_bytes)
        storage = al.make_shared((WORDS,), al.u32)
        route_view = al.subview(routes, (block * BM,), (BM,), (1,))
        route_resource = al.amdgpu.make_rsrc(route_view, BM * 4)
        metadata_row = lane % TB + wave * TB + (lane // TB) * BM
        storage[ARENA + metadata_row] = al.amdgpu.raw_buffer_load_x1(route_resource, metadata_row * 4, 0, 0)
        input_offsets = al.make_local((2,), al.u32)
        # Routing is invariant over the K loop. Keep the source row offsets
        # in registers so async copies do not reread the LDS metadata tail.
        for load in al.static_range(INPUT_LOADS):
            row = wave * TB + load * 8 + lane // 8
            token = storage[ARENA + row] & 0xFFFFFF
            input_offsets[load] = al.min(token, num_tokens) * (D // 2) + ((lane % 8) ^ (row & 7)) * 16
        act_resource = al.amdgpu.make_rsrc(ACT, num_tokens * D // 2)
        act_scale_resource = al.amdgpu.make_rsrc(ACT_SCALES, capacity * D // 32)
        stage1_compute(
            act_resource,
            act_scale_resource,
            resource_w,
            resource_ws,
            bias_resource,
            storage,
            input_offsets,
            block,
            wave,
            lane,
            tid,
        )
        hidden = al.view(storage, al.f32, al.make_layout((BM, BN // 4, 4), (BN, 4, 1)))
        output_routes = al.make_local((SLICES,), al.u32)
        if WM == 64:
            for batch in al.static_range(SLICES):
                output_routes[batch] = storage[ARENA + batch * ROWS_PER_SLICE + tid // COL_LANES]
        al.syncthreads()
        for batch in al.static_range(SLICES):
            row = batch * ROWS_PER_SLICE + tid // COL_LANES
            route = output_routes[batch] if WM == 64 else storage[ARENA + row]
            token, slot = route & 0xFFFFFF, route >> 24
            sorted_row = block * BM + row
            if sorted_row < route_extent and token < num_tokens and slot < TOPK:
                for segment in al.static_range(SEGMENTS):
                    col_lane = segment * 32 + tid % COL_LANES
                    values = al.make_local((4,), al.f32)
                    for c in al.static_range(4):
                        values[c] = hidden[row, col_lane, c]
                    act_row = sorted_row if SORTED else token * TOPK + slot
                    store_intermediate(
                        values,
                        output_resource,
                        al.convert(act_row, al.u32),
                        sorted_row,
                        al.convert(tile * BN, al.u32),
                        col_lane,
                        al.convert(0, al.u32),
                        al.convert(scale_base, al.u32),
                    )

    return stage1


@cache
def make_stage1(config: MoeConfig):
    """Resolve the selected solution and cache its local Stage1 kernel."""
    return resolve_2stage_implementation(config)[0](config)
