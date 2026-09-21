"""Compute W2 with reusable M32/N64 wave register tiles."""

from functools import cache

import avelang
import avelang.language as al

from .dispatch import resolve_2stage_implementation
from .intermediate_mxfp4 import make_stage2_input
from .weight_mxfp4 import load_weight_scales, load_weight_values, make_w2_resources


@cache
def make_stage2_compute(config):
    intermediate = config.intermediate
    K_TILES = config.intermediate // 256
    BIAS = config.bias
    prefetch_stage2_input, read_stage2_input = make_stage2_input(config)

    @avelang.jit
    def stage2_compute(
        act: al.Tensor((4,), al.u32),
        weight: al.Tensor((4,), al.u32),
        scales: al.Tensor((4,), al.u32),
        bias: al.Tensor((4,), al.u32),
        route_weights: al.Tensor((4,), al.u32),
        storage: al.Tensor((4160,), al.u32),
        act_offset: al.u32,
        input_valid: al.u1,
        block: al.u32,
        scale_base: al.u32,
        tile: al.u32,
        wave: al.u32,
        lane: al.u32,
    ):
        accum = al.full((4, 2, 4), 0, al.f32)
        fragments = al.make_local((2, 2, 4), al.u32)
        weights = al.make_local((2, 4, 4), al.u32)
        weight_scales = al.make_local((2,), al.u32)
        packed_bias = al.make_local((4, 2), al.u32)
        cached_route_weights = al.make_local((2,), al.f32)
        for m in al.static_range(2):
            bits = al.amdgpu.raw_buffer_load_x1(route_weights, (m * 16 + lane % 16) * 4, 0, 0)
            cached_route_weights[m] = al.bitcast(bits, al.f32)
        if BIAS:
            for n in al.static_range(4):
                col = tile * 256 + wave * 64 + n * 4 + lane // 16 * 16
                packed_bias[n] = al.amdgpu.raw_buffer_load_x2(bias, col * 2, 0, 0)
        input_row, vector = wave * 8 + lane // 8, lane % 8
        lds = al.view(storage, al.u32, al.make_layout((2, 32, 16, 4), (2048, 64, 4, 1)))
        for k in al.static_range(K_TILES):
            ki = al.convert(k, al.u32)
            prefetched, input_scale = prefetch_stage2_input(act, act_offset, input_valid, block, ki, lane, scale_base)
            lds[k % 2, input_row, vector ^ (input_row & 15)] = prefetched
            al.amdgpu.s_waitcnt(0, 0, 0)
            al.syncthreads()
            read_stage2_input(storage, fragments, ki, lane)
            for half_k in al.static_range(2):
                for n in al.static_range(4):
                    weights[half_k, n] = load_weight_values(weight, intermediate, wave * 4 + n, ki * 2 + half_k, lane)
            for n in al.static_range(2):
                weight_scales[n] = load_weight_scales(scales, intermediate, wave * 2 + n, ki, lane)
            al.amdgpu.s_waitcnt(0, 0, 0)
            for half_k in al.static_range(2):
                for n in al.static_range(4):
                    for m in al.static_range(2):
                        accum[n, m] = al.amdgpu.mfma_scale_16x16x128_fp4(
                            weights[half_k, n],
                            weight_scales[n // 2],
                            fragments[m, half_k],
                            input_scale,
                            accum[n, m],
                            2 * half_k + n % 2,
                            2 * half_k + m,
                        )
            if k + 1 < K_TILES:
                al.syncthreads()
        al.syncthreads()
        result = al.view(storage, al.bf16, al.make_layout((32, 256), (256, 1)))
        bias_values = al.view(packed_bias, al.bf16, al.make_layout((4, 4), (4, 1)))
        for m in al.static_range(2):
            for n in al.static_range(4):
                n16 = wave * 4 + n
                for c in al.static_range(4):
                    value = accum[n, m, c]
                    if BIAS:
                        value = value + al.convert(bias_values[n, c], al.f32)
                    result[m * 16 + lane % 16, n16 * 16 + lane // 16 * 4 + c] = al.convert(
                        value * cached_route_weights[m], al.bf16
                    )
        al.syncthreads()

    return stage2_compute


def make_stage2_kernel(config):
    D, I = config.hidden, config.intermediate
    E, TOPK = config.experts, config.topk
    WORKERS = config.stage2_workers
    RATIO = config.stage1_tile_m // 32
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
        tile, worker = al.convert(al.block_id(0), al.u32), al.convert(al.block_id(1), al.u32)
        tid = al.convert(al.thread_id(0), al.u32)
        wave, lane = al.amdgpu.readfirstlane(tid // 64), tid % 64
        extent, tokens = al.amdgpu.readfirstlane(counts[0]), al.amdgpu.readfirstlane(counts[1])
        storage = al.make_shared((4160,), al.u32)
        experts = al.make_tensor(expert_ptr, al.u32, al.make_layout((capacity // (32 * RATIO),), (1,)))
        groups = (extent + 31) // 32
        quotient, remainder = groups // WORKERS, groups % WORKERS
        begin = worker * quotient + al.min(worker, remainder)
        assigned = quotient + al.select(worker < remainder, al.convert(1, al.u32), al.convert(0, al.u32))
        for block in al.range(begin, begin + assigned):
            group = al.convert(block, al.u32)
            expert = al.amdgpu.readfirstlane(experts[group // RATIO])
            if expert < E:
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
                row_offsets = al.subview(storage, (4096,), (32,), (1,))
                if tid < 32:
                    metadata_route = al.amdgpu.raw_buffer_load_x1(route_resource, tid * 4, 0, 0)
                    metadata_token, metadata_slot = metadata_route & 0xFFFFFF, metadata_route >> 24
                    row_offsets[tid] = al.select(
                        group * 32 + tid < extent and metadata_token < tokens and metadata_slot < TOPK,
                        metadata_token * D * 2,
                        tokens * D * 2,
                    )
                input_row, vector = tid // 8, tid % 8
                input_route = al.amdgpu.raw_buffer_load_x1(route_resource, input_row * 4, 0, 0)
                input_token, input_slot = input_route & 0xFFFFFF, input_route >> 24
                input_valid = input_token < tokens and input_slot < TOPK
                act_offset = (input_token * TOPK + input_slot) * (I // 2) + vector * 16
                stage2_compute(
                    act_resource,
                    wr,
                    sr,
                    br,
                    rw_resource,
                    storage,
                    act_offset,
                    input_valid,
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
                    # Invalid rows start at the resource bound, so buffer atomics
                    # discard them without rereading route IDs or branching per pair.
                    offset = row_offsets[row] + tile * 512 + column * 2
                    al.amdgpu.raw_buffer_atomic_add_bf16x2(pairs[pair], output_resource, offset)
            # Protect the shared arena before this worker takes another group.
            al.syncthreads()

    return stage2


@cache
def make_stage2(config):
    return resolve_2stage_implementation(config)[1](config)
