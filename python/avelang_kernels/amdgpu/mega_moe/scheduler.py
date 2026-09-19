"""Find a tile by scanning destination expert counts."""

# ruff: noqa: PLR1716 -- The DSL requires explicit comparisons.

from functools import cache

import avelang
import avelang.language as al


@cache
def make_scheduler(layout, tile_m, tile_count):
    LE, BM, NT = layout.config.local_experts, tile_m, tile_count
    B, SLOT, SUM = layout.rank_sym_buffer_base, layout.rank_slot_bytes, layout.recv_sum

    @avelang.jit
    def get_work(
        resource: al.Tensor((4,), al.u32), rank: al.u32, logical: al.u32
    ) -> (al.u32, al.u32, al.u32, al.u32, al.u32):
        target, tile = logical // NT, logical % NT
        expert, pool, rows, found = (
            al.convert(0, al.u32),
            al.convert(0, al.u32),
            al.convert(0, al.u32),
            al.convert(0, al.u32),
        )
        logical_base, physical_base = al.convert(0, al.u32), al.convert(0, al.u32)
        for e in al.range(LE):
            count = al.amdgpu.raw_buffer_load_x1(resource, B + rank * SLOT + SUM + e * 8, 0, 17)
            tiles = (count + BM - 1) // BM
            if logical_base <= target and target < logical_base + tiles:
                expert = al.convert(e, al.u32)
                pool = physical_base + (target - logical_base) * BM
                rows = al.min(al.convert(BM, al.u32), count - (target - logical_base) * BM)
                found = al.convert(1, al.u32)
            logical_base = logical_base + tiles
            physical_base = physical_base + ((count + 31) // 32) * 32
        return expert, pool, rows, tile, found

    return get_work
