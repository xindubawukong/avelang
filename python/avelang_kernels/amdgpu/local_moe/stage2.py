"""K256 Stage2 compute and the local routed-output kernel.

Weight-as-A MFMA produces a tile in XOR-swizzled LDS. The wrapper owns task
selection, route validation and packed BF16 atomic output writes.
"""

from functools import cache

import avelang
import avelang.language as al

from .config import MoeConfig
from .dispatch import resolve_2stage_implementation
from .intermediate_mxfp4 import make_stage2_input_k256
from .solutionid import DataType
from .weight_mxfp4 import make_w2_resources

STAGE2_K256_ARENA_WORDS = 4096
STAGE2_K256_LDS_WORDS = 4160


@avelang.jit
def output_word_index(row: al.u32, column: al.u32) -> al.u32:
    """Map a BF16 column in the N256 output tile to an XOR-swizzled u32 word."""
    return ((row * 256 + column) // 2) ^ ((row & 15) * 4)


@avelang.jit
def _pack_weighted_bf16_pair(first: al.f32, second: al.f32, route_weight: al.f32) -> al.u32:
    """Pack one pair without passing the complete accumulator through a helper."""
    values = al.make_local((1, 2), al.f32)
    values[0, 0] = first * route_weight
    values[0, 1] = second * route_weight
    packed = al.make_local((1, 2), al.bf16)
    packed[0] = al.convert(values[0], al.bf16)
    # Two BF16 values occupy one u32; a one-element view is scalar.
    word = al.view(packed, al.u32, al.make_layout((1,), (1,)))
    return word


@cache
def make_stage2_compute_k256(intermediate, bias, weight_cache, act_cache=0, words=STAGE2_K256_LDS_WORDS):
    """Write weighted BF16 [32, 256] output into swizzled LDS and synchronize."""
    I, BIAS, K_TILES = intermediate, bias, intermediate // 256
    prefetch_stage2_input, read_stage2_input = make_stage2_input_k256(words, act_cache)

    @avelang.jit
    def stage2_compute_k256(
        act_resource: al.Tensor((4,), al.u32),
        weight_resource: al.Tensor((4,), al.u32),
        scale_resource: al.Tensor((4,), al.u32),
        bias_resource: al.Tensor((4,), al.u32),
        rw_resource: al.Tensor((4,), al.u32),
        storage: al.Tensor((words,), al.u32),
        act_offset: al.u32,
        act_base: al.u32,
        scale_offset: al.u32,
        scale_base: al.u32,
        input_valid: al.u1,
        work_m: al.u32,
        tile: al.u32,
        tid: al.u32,
    ):
        lane, wave = tid % 64, al.amdgpu.readfirstlane(tid // 64)
        input_row, vector = tid // 8, tid % 8
        accum = al.full((4, 2, 4), 0, al.f32)
        lds = al.view(storage, al.u32, al.make_layout((2, 32, 16, 4), (2048, 64, 4, 1)))
        weights = al.make_local((2, 4, 4), al.u32)
        weight_scales = al.make_local((2,), al.u32)
        fragments = al.make_local((2, 2, 4), al.u32)
        packed_bias = al.make_local((4, 2), al.u32)
        route_weights = al.make_local((2,), al.f32)
        for m in al.static_range(2):
            bits = al.amdgpu.raw_buffer_load_x1(rw_resource, (m * 16 + lane % 16) * 4, 0, act_cache)
            route_weights[m] = al.bitcast(bits, al.f32)
        if BIAS:
            for n in al.static_range(4):
                offset = (tile * 256 + wave * 64 + n * 4 + (lane // 16) * 16) * 2
                packed_bias[n] = al.amdgpu.raw_buffer_load_x2(bias_resource, offset, 0, 0)
        for k in al.static_range(K_TILES):
            stage = k % 2
            prefetched, scale = prefetch_stage2_input(
                act_resource, act_offset, act_base, scale_offset, scale_base, input_valid, al.convert(k, al.u32)
            )
            # Keep the weight issue group next to the LDS handoff/MFMA loop.
            # A bulk-load helper changed allocation/scheduling and slowed DSv4 EP8.
            for half_k in al.static_range(2):
                for n in al.static_range(4):
                    offset_w = (wave * 64 + n * 16) * (I // 2) + half_k * 1024 + lane * 16
                    weights[half_k, n] = al.amdgpu.raw_buffer_load_x4(weight_resource, offset_w, k * 2048, weight_cache)
            for n in al.static_range(2):
                weight_scales[n] = al.amdgpu.raw_buffer_load_x1(
                    scale_resource, (wave * 2 + n) * I + lane * 4, k * 256, 0
                )
            lds[stage, input_row, vector ^ (input_row & 15)] = prefetched
            al.syncthreads()
            read_stage2_input(storage, fragments, al.convert(k, al.u32), lane)
            for half_k in al.static_range(2):
                for n in al.static_range(4):
                    for m in al.static_range(2):
                        accum[n, m] = al.amdgpu.mfma_scale_16x16x128_fp4(
                            weights[half_k, n],
                            weight_scales[n // 2],
                            fragments[m, half_k],
                            scale,
                            accum[n, m],
                            2 * half_k + n % 2,
                            2 * half_k + m,
                        )
        al.syncthreads()
        # Keep accumulator traversal here. Passing the entire tile to a helper
        # raised GPT-OSS VGPR usage from 128 to 132 and regressed Stage2 timing.
        bias_values = al.view(packed_bias, al.bf16, al.make_layout((4, 4), (4, 1)))
        pairs = al.make_local((2,), al.u32)
        for m in al.static_range(2):
            row = m * 16 + lane % 16
            if row < work_m:
                for n in al.static_range(4):
                    for pair in al.static_range(2):
                        first, second = accum[n, m, pair * 2], accum[n, m, pair * 2 + 1]
                        if BIAS:
                            first = first + al.convert(bias_values[n, pair * 2], al.f32)
                            second = second + al.convert(bias_values[n, pair * 2 + 1], al.f32)
                        pairs[pair] = _pack_weighted_bf16_pair(first, second, route_weights[m])
                    index = output_word_index(row, wave * 64 + n * 16 + (lane // 16) * 4)
                    storage[index] = pairs[0]
                    storage[index + 1] = pairs[1]
        al.syncthreads()

    return stage2_compute_k256


def make_stage2_kernel(config: MoeConfig):
    """Build the local Stage2 kernel with persistent K256 workgroups."""
    D, I, E, TOPK = config.hidden, config.intermediate, config.experts, config.topk
    BIAS = config.solution.bias_dtype == DataType.BF16
    WORKERS = config.stage2_workers
    WEIGHT_CACHE = config.stage2_weight_load_aux
    RATIO = config.stage1_tile_m // 32
    WORDS, ROW_OFFSETS = STAGE2_K256_LDS_WORDS, STAGE2_K256_ARENA_WORDS
    stage2_compute_k256 = make_stage2_compute_k256(I, BIAS, WEIGHT_CACHE)
    initialize_w2_resources = make_w2_resources(D, I, E, I // 32)

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
        tile, worker = al.block_id(0), al.block_id(1)
        tid = al.thread_id(0)
        lane = tid % 64
        wave = al.amdgpu.readfirstlane(al.convert(tid // 64, al.u32))
        groups = (counts[0] + 31) // 32
        quotient, remainder = groups // WORKERS, groups % WORKERS
        begin = worker * quotient + al.min(worker, remainder)
        assigned = quotient + al.select(worker < remainder, al.convert(1, al.u32), al.convert(0, al.u32))
        for block in al.range(begin, begin + assigned):
            experts = al.make_tensor(expert_ptr, al.u32, al.make_layout((capacity // (32 * RATIO),), (1,)))
            expert = al.amdgpu.readfirstlane(experts[block // RATIO])
            if expert < E:
                routes = al.make_tensor(ids_ptr, al.u32, al.make_layout((capacity,), (1,)))
                route_weights = al.make_tensor(route_weight_ptr, al.f32, al.make_layout((capacity,), (1,)))
                act_bytes = capacity * I // 2
                workspace_bytes = act_bytes + ((capacity + 255) // 256) * 256 * (I // 32)
                workspace = al.make_tensor(workspace_ptr, al.u8, al.make_layout((workspace_bytes,), (1,)))
                act_resource = al.amdgpu.make_rsrc(workspace, workspace_bytes)
                weight_resource, scale_resource, bias_resource = initialize_w2_resources(
                    weight, ws, bias_ptr, expert, al.convert(tile, al.u32), al.convert(1, al.u32)
                )
                output = al.make_tensor(out_ptr, al.bf16, al.make_layout((counts[1] * D,), (1,)))
                output_resource = al.amdgpu.make_rsrc(output, counts[1] * D * 2)
                r_view = al.subview(routes, (block * 32,), (32,), (1,))
                rw_view = al.subview(route_weights, (block * 32,), (32,), (1,))
                route_resource = al.amdgpu.make_rsrc(r_view, 128)
                rw_resource = al.amdgpu.make_rsrc(rw_view, 128)

                storage = al.make_shared((WORDS,), al.u32)
                row_offsets = al.subview(storage, (ROW_OFFSETS,), (32,), (1,))
                input_row, vector = tid // 8, tid % 8
                route = al.amdgpu.raw_buffer_load_x1(route_resource, al.convert(input_row * 4, al.u32), 0, 0)
                token, slot = route & 0xFFFFFF, route >> 24
                valid = token < counts[1] and slot < TOPK
                if tid < 32:
                    metadata_route = al.amdgpu.raw_buffer_load_x1(route_resource, al.convert(tid * 4, al.u32), 0, 0)
                    metadata_token, metadata_slot = metadata_route & 0xFFFFFF, metadata_route >> 24
                    row_offsets[tid] = al.select(
                        metadata_token < counts[1] and metadata_slot < TOPK, metadata_token * D * 2, counts[1] * D * 2
                    )
                stage2_compute_k256(
                    act_resource,
                    weight_resource,
                    scale_resource,
                    bias_resource,
                    rw_resource,
                    storage,
                    al.convert((token * TOPK + slot) * (I // 2) + vector * 16, al.u32),
                    al.convert(0, al.u32),
                    al.convert(block * 32 * (I // 32) + lane * 4, al.u32),
                    al.convert(act_bytes, al.u32),
                    valid,
                    al.convert(32, al.u32),
                    al.convert(tile, al.u32),
                    al.convert(tid, al.u32),
                )
                shared = al.view(storage, al.bf16, al.make_layout((4096, 2), (2, 1)))
                # Counts are invariant during this kernel. Snapshot the bound
                # here so each atomic row does not reload counts and wait on VMEM.
                # Keeping its lifetime after the GEMM also avoids increasing VGPRs.
                output_bytes = al.amdgpu.readfirstlane(counts[1]) * D * 2
                for m in al.static_range(8):
                    row = wave * 8 + m
                    row_offset = al.amdgpu.readfirstlane(row_offsets[row])
                    # Skip invalid output rows.
                    if row_offset < output_bytes:
                        offset = al.convert(row_offset + tile * 512 + lane * 4, al.u32)
                        index = output_word_index(row, lane * 2)
                        al.amdgpu.raw_buffer_atomic_add_bf16x2(shared[index], output_resource, offset)
                        al.amdgpu.raw_buffer_atomic_add_bf16x2(shared[index + 64], output_resource, offset + 256)
            # Protect the shared arena before this worker takes its next route tile.
            al.syncthreads()

    return stage2


@cache
def make_stage2(config: MoeConfig):
    """Resolve the selected solution and cache its local Stage2 kernel."""
    factory = resolve_2stage_implementation(config)[1]
    return factory(config)
