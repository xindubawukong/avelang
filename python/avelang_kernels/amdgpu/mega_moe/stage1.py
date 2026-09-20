"""Stage1 tile execution and the persistent dispatch/task loop."""

from functools import cache

import avelang
import avelang.language as al

from ..local_moe.intermediate_mxfp4 import make_intermediate_store
from ..local_moe.stage1 import make_stage1_compute
from ..local_moe.weight_mxfp4 import make_w13_resources
from .input_mxfp4_packed import make_mxfp4_packed_input
from .scheduler import make_scheduler
from .synchronization import complete_stores, wait_mask
from .token_shuffle_direct_push import make_direct_push_token_shuffle
from .workspace import WorkspaceLayout


@cache
def make_stage1_tile(config):
    """Build a device helper to wait for, compute and publish one Stage1 tile."""
    layout = WorkspaceLayout(config)
    s = config.solution
    D, I, E = config.compute_hidden, s.intermediate, config.local_experts
    BM, BN, THREADS = config.stage1_tile_m, config.stage1_projection_n, config.stage1_num_warps * 64
    WORDS, ROW_BYTES = config.stage1_lds_words, config.input_token_bytes
    TB, LOADS = BM // config.stage1_num_warps, BM // config.stage1_num_warps // 8
    B, SLOT, L1, READY = layout.rank_sym_buffer_base, layout.rank_slot_bytes, layout.l1_tokens, layout.l1_ready
    L2, SCALES, L2_READY = layout.l2_tokens, layout.l2_scales, layout.l2_ready
    SCALE_COLS = layout.scale_cols
    ROWS_PER_SLICE, SLICES = THREADS // 32, BM // (THREADS // 32)
    prepare_scales, prefetch_input, read_input = make_mxfp4_packed_input(config, s.hidden, ROW_BYTES, 17)
    stage1_compute = make_stage1_compute(config, prefetch_input=prefetch_input, read_input=read_input)
    initialize_w13_resources = make_w13_resources(D, I, E, BN, I)
    store_intermediate = make_intermediate_store(I, SCALE_COLS, act_aux=17, scale_aux=17)

    @avelang.jit
    def run_stage1_tile(
        resource: al.Tensor((4,), al.u32),
        storage: al.Tensor((WORDS,), al.u32),
        w: al.Pointer(al.u32),
        ws: al.Pointer(al.u32),
        bias: al.Pointer(al.bf16),
        expert: al.u32,
        pool: al.u32,
        rows: al.u32,
        tile: al.u32,
        rank: al.u32,
        bias_enabled: al.u32,
        tid: al.u32,
    ):
        wave, lane = al.amdgpu.readfirstlane(tid // 64), tid % 64
        if tid == 0:
            for subblock in al.range((rows + 31) // 32):
                nrows = al.min(al.convert(32, al.u32), rows - subblock * 32)
                mask = al.select(nrows == 32, al.convert(0xFFFFFFFF, al.u32), (1 << nrows) - 1)
                wait_mask(resource, B + rank * SLOT + READY + (pool // 32 + subblock) * 4, mask)
        al.syncthreads()
        al.amdgpu.compiler_barrier()
        al.syncthreads()
        row_base = B + rank * SLOT + L1 + pool * ROW_BYTES
        prepare_scales(resource, storage, row_base, tid)
        input_offsets = al.make_local((2,), al.u32)
        for load in al.static_range(LOADS):
            row = wave * TB + load * 8 + lane // 8
            vec = (lane % 8) ^ (row & 7)
            input_offsets[load] = al.select(
                row < rows, row_base + row * ROW_BYTES + vec * 16, al.convert(0xFFFFFFFF, al.u32)
            )
        wr, sr, br = initialize_w13_resources(w, ws, bias, expert, tile, bias_enabled)
        stage1_compute(resource, resource, wr, sr, br, storage, input_offsets, pool // BM, wave, lane)
        hidden = al.view(storage, al.f32, al.make_layout((BM, BN // 4, 4), (BN, 4, 1)))
        al.syncthreads()
        for batch in al.static_range(SLICES):
            row = batch * ROWS_PER_SLICE + tid // 32
            if row < rows:
                for segment in al.static_range(2):
                    col_lane = segment * 32 + tid % 32
                    values = hidden[row, col_lane]
                    store_intermediate(
                        values,
                        resource,
                        pool + row,
                        pool + row,
                        tile * BN,
                        col_lane,
                        al.convert(L2, al.u32),
                        al.convert(SCALES, al.u32),
                    )
        complete_stores()
        al.syncthreads()
        if tid == 0:
            for subblock in al.range((rows + 31) // 32):
                al.amdgpu.raw_buffer_atomic_or_u32(1 << tile, resource, L2_READY + (pool // 32 + subblock) * 4, 0, 16)
        al.syncthreads()

    return run_stage1_tile


@cache
def make_stage1(config):
    """Build the kernel that dispatches tokens and repeatedly runs Stage1 tiles."""
    layout = WorkspaceLayout(config)
    I, TOPK = config.solution.intermediate, config.solution.topk
    BM, THREADS = config.stage1_tile_m, config.stage1_num_warps * 64
    WORDS, SIZE, ROW_BYTES = config.stage1_lds_words, layout.workspace_bytes, config.input_token_bytes
    HEADS = layout.work_heads
    direct_push_token_shuffle = make_direct_push_token_shuffle(layout, THREADS, WORDS)
    load_expert_metadata, get_work = make_scheduler(layout, BM, I // 256)
    run_stage1_tile = make_stage1_tile(config)

    @avelang.jit
    def stage1(
        heap: al.Pointer(al.u8),
        x: al.Pointer(al.u8),
        ids: al.Pointer(al.u32),
        rw: al.Pointer(al.f32),
        w: al.Pointer(al.u32),
        ws: al.Pointer(al.u32),
        bias: al.Pointer(al.bf16),
        count: al.u32,
        rank: al.u32,
        bias_enabled: al.u32,
    ):
        tid, block = al.convert(al.thread_id(0), al.u32), al.convert(al.block_id(0), al.u32)
        lane = tid % 64
        memory = al.make_tensor(heap, al.u8, al.make_layout((SIZE,), (1,)))
        resource = al.amdgpu.make_rsrc(memory, SIZE)
        inputs = al.make_tensor(x, al.u8, al.make_layout((count * ROW_BYTES,), (1,)))
        routes = al.make_tensor(ids, al.u32, al.make_layout((count * TOPK,), (1,)))
        weights = al.make_tensor(rw, al.f32, al.make_layout((count * TOPK,), (1,)))
        xr = al.amdgpu.make_rsrc(inputs, count * ROW_BYTES)
        ir = al.amdgpu.make_rsrc(routes, count * TOPK * 4)
        rr = al.amdgpu.make_rsrc(weights, count * TOPK * 4)
        storage = al.make_shared((WORDS,), al.u32)
        direct_push_token_shuffle(resource, storage, xr, ir, rr, count, rank, block, tid)
        lane_tokens, lane_base = load_expert_metadata(resource, rank, lane)
        active = al.convert(1, al.u32)
        while active != 0:
            if tid == 0:
                shard = block % 8
                ticket = al.amdgpu.raw_buffer_atomic_add_u32(al.convert(1, al.u32), resource, HEADS + shard * 64, 0, 16)
                storage[WORDS - 1] = shard + ticket * 8
            al.syncthreads()
            logical = storage[WORDS - 1]
            expert, pool, rows, tile, found = get_work(lane_tokens, lane_base, logical, lane)
            active = al.amdgpu.readfirstlane(found)
            if active != 0:
                expert, pool = al.amdgpu.readfirstlane(expert), al.amdgpu.readfirstlane(pool)
                rows, tile = al.amdgpu.readfirstlane(rows), al.amdgpu.readfirstlane(tile)
                run_stage1_tile(resource, storage, w, ws, bias, expert, pool, rows, tile, rank, bias_enabled, tid)

    return stage1
