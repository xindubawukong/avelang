"""Persistent tickets over expert runs padded to M32 transport rows."""

# ruff: noqa: PLR1716 -- Keep explicit comparisons in DSL control flow.

from functools import cache

import avelang
import avelang.language as al


@cache
def make_scheduler(layout, tile_m, tile_count):
    LE = layout.config.local_experts
    B, SLOT, SUM = layout.rank_sym_buffer_base, layout.rank_slot_bytes, layout.recv_sum

    @avelang.jit
    def load_expert_metadata(resource: al.Tensor((4,), al.u32), rank: al.u32, lane: al.u32) -> (al.u32, al.u32):
        tokens = al.convert(0, al.u32)
        if lane < LE:
            tokens = al.amdgpu.raw_buffer_load_x1(resource, B + rank * SLOT + SUM + lane * 8, 0, 16)
        blocks = (tokens + 31) // 32
        inclusive = blocks
        for shift in al.static_range(6):
            peer = al.shuffle_up(inclusive, 1 << shift, 64)
            inclusive = inclusive + al.select(lane >= (1 << shift), peer, al.convert(0, al.u32))
        return tokens, inclusive - blocks

    @avelang.jit
    def get_work(
        tokens: al.u32, base: al.u32, logical: al.u32, lane: al.u32
    ) -> (al.u32, al.u32, al.u32, al.u32, al.u32):
        target, tile = logical // tile_count, logical % tile_count
        expert, pool, rows, found = (
            al.convert(0, al.u32),
            al.convert(0, al.u32),
            al.convert(0, al.u32),
            al.convert(0, al.u32),
        )
        if tile_m == 32:
            candidate = al.select(
                (lane < LE) and (base <= target) and (target < base + (tokens + 31) // 32), lane, al.convert(64, al.u32)
            )
            for shift in al.static_range(6):
                candidate = al.min(candidate, al.shuffle_xor(candidate, 1 << shift, 64))
            if candidate < LE:
                expert = candidate
                count = al.shuffle(tokens, candidate, 64)
                expert_base = al.shuffle(base, candidate, 64)
                pool = target * 32
                rows = al.min(al.convert(32, al.u32), count - (target - expert_base) * 32)
                found = al.convert(1, al.u32)
        else:
            logical_base, physical_base = al.convert(0, al.u32), al.convert(0, al.u32)
            for e in al.range(LE):
                count = al.shuffle(tokens, e, 64)
                blocks = (count + tile_m - 1) // tile_m
                if logical_base <= target and target < logical_base + blocks:
                    expert = al.convert(e, al.u32)
                    pool = physical_base * 32 + (target - logical_base) * tile_m
                    rows = al.min(al.convert(tile_m, al.u32), count - (target - logical_base) * tile_m)
                    found = al.convert(1, al.u32)
                logical_base = logical_base + blocks
                physical_base = physical_base + (count + 31) // 32
        return expert, pool, rows, tile, found

    return load_expert_metadata, get_work
