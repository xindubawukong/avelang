"""Stage2 tile execution and the persistent task loop."""

from functools import cache

import avelang
import avelang.language as al

from ..local_moe.stage2 import STAGE2_K256_LDS_WORDS, make_stage2_compute_k256, output_word_index
from ..local_moe.weight_mxfp4 import make_w2_resources
from .scheduler import make_scheduler
from .solutionid import DataType
from .synchronization import wait_mask
from .workspace import WorkspaceLayout


@cache
def make_stage2_tile(config):
    """Build a device helper to consume one intermediate tile and write route outputs."""
    layout = WorkspaceLayout(config)
    s = config.solution
    D, I, E, LOGICAL = config.compute_hidden, s.intermediate, config.local_experts, s.hidden
    SIZE, B, SLOT = layout.workspace_bytes, layout.rank_sym_buffer_base, layout.rank_slot_bytes
    META, RW, OUT = layout.metadata, layout.l1_weights, layout.route_output
    L2, SCALES, READY = layout.l2_tokens, layout.l2_scales, layout.l2_ready
    MASK = (1 << (I // 256)) - 1
    WORDS = STAGE2_K256_LDS_WORDS
    stage2_compute_k256 = make_stage2_compute_k256(I, s.bias_dtype == DataType.BF16, 2, 17)
    initialize_w2_resources = make_w2_resources(D, I, E, I // 32)

    @avelang.jit
    def run_stage2_tile(
        heap: al.Pointer(al.u8),
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
        memory = al.make_tensor(heap, al.u8, al.make_layout((SIZE,), (1,)))
        if tid == 0:
            wait_mask(resource, al.convert(READY + (pool // 32) * 4, al.u32), al.convert(MASK, al.u32))
        al.syncthreads()
        wr, sr, br = initialize_w2_resources(w, ws, bias, expert, tile, bias_enabled)
        rw_offset = al.convert(rank, al.u64) * SLOT + B + RW + al.convert(pool, al.u64) * 4
        rwv = al.subview(memory, (rw_offset,), (128,), (1,))
        rr = al.amdgpu.make_rsrc(rwv, 128)
        stage2_compute_k256(
            resource,
            wr,
            sr,
            br,
            rr,
            storage,
            (pool + tid // 8) * (I // 2) + (tid % 8) * 16,
            al.convert(L2, al.u32),
            pool * (I // 32) + lane * 4,
            al.convert(SCALES, al.u32),
            tid // 8 < rows,
            rows,
            tile,
            tid,
        )
        for group in al.static_range(8):
            row = wave + group * 4
            if row < rows:
                metadata = al.amdgpu.raw_buffer_load_x2(resource, B + rank * SLOT + META + (pool + row) * 8, 0, 17)
                route = al.amdgpu.readfirstlane(al.convert(metadata[0], al.u32))
                source = al.amdgpu.readfirstlane(al.convert(metadata[1], al.u32))
                row_view = al.subview(
                    memory,
                    (al.convert(source, al.u64) * SLOT + B + OUT + al.convert(route, al.u64) * (LOGICAL * 2),),
                    (LOGICAL * 2,),
                    (1,),
                )
                output_resource = al.amdgpu.make_rsrc(row_view, LOGICAL * 2)
                for half in al.static_range(2):
                    value = storage[output_word_index(row, (half * 64 + lane) * 2)]
                    al.amdgpu.raw_buffer_store_x1(value, output_resource, tile * 512 + (half * 64 + lane) * 4, 0, 17)
        al.amdgpu.fence(1, 2)

        al.syncthreads()

    return run_stage2_tile


@cache
def make_stage2(config):
    """Build the kernel that repeatedly runs Stage2 tiles."""
    layout = WorkspaceLayout(config)
    D = config.compute_hidden
    SIZE, WORDS = layout.workspace_bytes, STAGE2_K256_LDS_WORDS
    load_expert_metadata, get_work = make_scheduler(layout, 32, D // 256)
    run_stage2_tile = make_stage2_tile(config)

    @avelang.jit
    def stage2(
        heap: al.Pointer(al.u8),
        w: al.Pointer(al.u32),
        ws: al.Pointer(al.u32),
        bias: al.Pointer(al.bf16),
        rank: al.u32,
        bias_enabled: al.u32,
    ):
        tid = al.convert(al.thread_id(0), al.u32)
        lane = tid % 64
        memory = al.make_tensor(heap, al.u8, al.make_layout((SIZE,), (1,)))
        resource = al.amdgpu.make_rsrc(memory, SIZE)
        storage = al.make_shared((WORDS,), al.u32)
        lane_tokens, lane_base = load_expert_metadata(resource, rank, lane)
        logical, active = al.convert(al.block_id(0), al.u32), al.convert(1, al.u32)
        while active != 0:
            expert, pool, rows, tile, found = get_work(lane_tokens, lane_base, logical, lane)
            active = al.amdgpu.readfirstlane(found)
            if active != 0:
                expert, pool = al.amdgpu.readfirstlane(expert), al.amdgpu.readfirstlane(pool)
                rows, tile = al.amdgpu.readfirstlane(rows), al.amdgpu.readfirstlane(tile)
                run_stage2_tile(heap, resource, storage, w, ws, bias, expert, pool, rows, tile, rank, bias_enabled, tid)
                logical = logical + 256

    return stage2
