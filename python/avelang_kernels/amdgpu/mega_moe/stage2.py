"""Reuse Stage2 workgroups across tiles and return route rows to their sources."""

from functools import cache

import avelang
import avelang.language as al

from ..local_moe.stage2 import STAGE2_K256_LDS_WORDS, make_stage2_compute_k256, output_word_index
from ..local_moe.weight_mxfp4 import make_w2_resources
from .scheduler import make_scheduler
from .workspace import WorkspaceLayout


@cache
def make_stage2(config):
    layout = WorkspaceLayout(config)
    I, D, LOGICAL, E = config.intermediate, config.compute_hidden, config.solution.hidden, config.local_experts
    SIZE, B, SLOT = layout.workspace_bytes, layout.rank_sym_buffer_base, layout.rank_slot_bytes
    ACT, SCALES, SC = layout.l2_tokens, layout.l2_scales, layout.scale_cols
    RW, META, OUT = layout.l1_weights, layout.metadata, layout.route_output
    WORDS = STAGE2_K256_LDS_WORDS
    get_work = make_scheduler(layout, 32, D // 256)
    initialize_w2_resources = make_w2_resources(D, I, E, SC)
    stage2_compute_k256 = make_stage2_compute_k256(I, config.bias, 2, act_cache=17)

    @avelang.jit
    def stage2(
        heap: al.Pointer(al.u8),
        weight: al.Pointer(al.u32),
        scales: al.Pointer(al.u32),
        bias: al.Pointer(al.bf16),
        rank: al.u32,
        bias_enabled: al.u32,
    ):
        tid, logical = al.convert(al.thread_id(0), al.u32), al.convert(al.block_id(0), al.u32)
        lane = tid % 64
        memory = al.make_tensor(heap, al.u8, al.make_layout((SIZE,), (1,)))
        resource = al.amdgpu.make_rsrc(memory, SIZE)
        storage = al.make_shared((WORDS,), al.u32)
        active = al.convert(1, al.u32)
        while active != 0:
            expert, pool, rows, tile, found = get_work(resource, rank, logical)
            active = al.amdgpu.readfirstlane(found)
            if active != 0:
                wr, sr, br = initialize_w2_resources(weight, scales, bias, expert, tile, bias_enabled)
                rw_offset = al.convert(rank, al.u64) * SLOT + B + RW + al.convert(pool, al.u64) * 4
                weight_view = al.subview(memory, (rw_offset,), (128,), (1,))
                rw_resource = al.amdgpu.make_rsrc(weight_view, 128)
                input_row, vector = tid // 8, tid % 8
                stage2_compute_k256(
                    resource,
                    wr,
                    sr,
                    br,
                    rw_resource,
                    storage,
                    (pool + input_row) * (I // 2) + vector * 16,
                    al.convert(ACT, al.u32),
                    pool * SC + lane * 4,
                    al.convert(SCALES, al.u32),
                    input_row < rows,
                    rows,
                    tile,
                    tid,
                )
                for word in al.static_range(16):
                    linear = word * 256 + tid
                    row, column = linear // 128, (linear % 128) * 2
                    if row < rows and tile * 256 + column < LOGICAL:
                        metadata = al.amdgpu.raw_buffer_load_x2(
                            resource, B + rank * SLOT + META + (pool + row) * 8, 0, 17
                        )
                        route, source = metadata[0], metadata[1]
                        value = storage[output_word_index(row, column)]
                        offset = B + source * SLOT + OUT + route * LOGICAL * 2 + tile * 512 + column * 2
                        al.amdgpu.raw_buffer_store_x1(value, resource, offset, 0, 17)
                al.amdgpu.fence(1, 2)
                al.syncthreads()
                logical = logical + 256

    return stage2
