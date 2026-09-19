"""Persistent tickets over expert runs padded to M32 transport rows."""

# ruff: noqa: PLR1716 -- Keep explicit comparisons in DSL control flow.

from functools import cache

import avelang
import avelang.language as al


@cache
def make_scheduler(layout, tile_m, tile_count):
    """Build Petit's one-wave scheduler for up to two experts per lane."""
    LE = layout.config.local_experts
    if LE > 128:
        raise ValueError("MegaMoE scheduler supports at most 128 local experts")
    SUM = layout.recv_sum

    if LE <= 64:

        @avelang.jit
        def load_expert_metadata(resource: al.Tensor((4,), al.u32), rank: al.u32, lane: al.u32) -> (al.u32, al.u32):
            tokens = al.convert(0, al.u32)
            if lane < LE:
                tokens = al.amdgpu.raw_buffer_load_x1(resource, SUM + lane * 8, 0, 16)
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
                blocks = (tokens + 31) // 32
                candidate = al.select(
                    (lane < LE) and (base <= target) and (target < base + blocks),
                    lane,
                    al.convert(64, al.u32),
                )
                for shift in al.static_range(6):
                    candidate = al.min(candidate, al.shuffle_xor(candidate, 1 << shift, 64))
                if candidate < LE:
                    count = al.shuffle(tokens, candidate, 64)
                    expert_base = al.shuffle(base, candidate, 64)
                    expert = candidate
                    pool = target * 32
                    rows = al.min(al.convert(32, al.u32), count - (target - expert_base) * 32)
                    found = al.convert(1, al.u32)
            else:
                logical_base, physical_base = al.convert(0, al.u32), al.convert(0, al.u32)
                for current in al.range(LE):
                    count = al.shuffle(tokens, current, 64)
                    blocks = (count + tile_m - 1) // tile_m
                    if logical_base <= target and target < logical_base + blocks:
                        expert = current
                        pool = physical_base * 32 + (target - logical_base) * tile_m
                        rows = al.min(al.convert(tile_m, al.u32), count - (target - logical_base) * tile_m)
                        found = al.convert(1, al.u32)
                    logical_base = logical_base + blocks
                    physical_base = physical_base + (count + 31) // 32
            return expert, pool, rows, tile, found

        return load_expert_metadata, get_work

    experts_per_lane = 2

    @avelang.jit
    def load_expert_metadata(
        resource: al.Tensor((4,), al.u32), rank: al.u32, lane: al.u32
    ) -> (al.Tensor((experts_per_lane,), al.u32), al.Tensor((experts_per_lane,), al.u32)):
        tokens = al.make_local((experts_per_lane,), al.u32)
        bases = al.make_local((experts_per_lane,), al.u32)
        preceding = al.convert(0, al.u32)
        for bank in al.static_range(experts_per_lane):
            expert = bank * 64 + lane
            count = al.convert(0, al.u32)
            if expert < LE:
                count = al.amdgpu.raw_buffer_load_x1(resource, SUM + expert * 8, 0, 16)
            blocks = (count + 31) // 32
            inclusive = blocks
            for shift in al.static_range(6):
                peer = al.shuffle_up(inclusive, 1 << shift, 64)
                inclusive = inclusive + al.select(lane >= (1 << shift), peer, al.convert(0, al.u32))
            tokens[bank] = count
            bases[bank] = preceding + inclusive - blocks
            preceding = preceding + al.shuffle(inclusive, 63, 64)
        return tokens, bases

    @avelang.jit
    def get_work(
        tokens: al.Tensor((experts_per_lane,), al.u32),
        bases: al.Tensor((experts_per_lane,), al.u32),
        logical: al.u32,
        lane: al.u32,
    ) -> (al.u32, al.u32, al.u32, al.u32, al.u32):
        target, tile = logical // tile_count, logical % tile_count
        expert, pool, rows, found = (
            al.convert(0, al.u32),
            al.convert(0, al.u32),
            al.convert(0, al.u32),
            al.convert(0, al.u32),
        )
        if tile_m == 32:
            for bank in al.static_range(experts_per_lane):
                blocks = (tokens[bank] + 31) // 32
                candidate = al.select(
                    (bank * 64 + lane < LE) and (bases[bank] <= target) and (target < bases[bank] + blocks),
                    lane,
                    al.convert(64, al.u32),
                )
                for shift in al.static_range(6):
                    candidate = al.min(candidate, al.shuffle_xor(candidate, 1 << shift, 64))
                if found == 0 and candidate < 64 and bank * 64 + candidate < LE:
                    count = al.shuffle(tokens[bank], candidate, 64)
                    expert_base = al.shuffle(bases[bank], candidate, 64)
                    expert = bank * 64 + candidate
                    pool = target * 32
                    rows = al.min(al.convert(32, al.u32), count - (target - expert_base) * 32)
                    found = al.convert(1, al.u32)
        else:
            logical_base, physical_base = al.convert(0, al.u32), al.convert(0, al.u32)
            for current in al.range(LE):
                lane_value = al.select(current < 64, tokens[0], tokens[1])
                count = al.shuffle(lane_value, current % 64, 64)
                blocks = (count + tile_m - 1) // tile_m
                if logical_base <= target and target < logical_base + blocks:
                    expert = current
                    pool = physical_base * 32 + (target - logical_base) * tile_m
                    rows = al.min(al.convert(tile_m, al.u32), count - (target - logical_base) * tile_m)
                    found = al.convert(1, al.u32)
                logical_base = logical_base + blocks
                physical_base = physical_base + (count + 31) // 32
        return expert, pool, rows, tile, found

    return load_expert_metadata, get_work
