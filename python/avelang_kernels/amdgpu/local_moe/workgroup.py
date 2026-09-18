"""Balanced grouped workgroup mapping shared by the two local MoE stages."""

from functools import cache

import avelang
import avelang.language as al


@cache
def make_grouped_workgroup_mapping(n_tiles: int, m_group: int, groups: int):
    NT, MG, GROUPS = (n_tiles, m_group, groups)

    @avelang.jit
    def map_workgroup(block: al.u32, grid_m: al.u32) -> (al.u32, al.u32):
        blocks = grid_m * NT
        group = block % GROUPS
        per_group, extra = (blocks // GROUPS, blocks % GROUPS)
        remapped = group * per_group + al.min(group, extra) + block // GROUPS
        first_m = remapped // (MG * NT) * MG
        group_m = al.min(grid_m - first_m, MG)
        in_group = remapped % (MG * NT)
        tile_n = al.convert(0, al.u32)
        tile_m = al.convert(0, al.u32)
        if MG == 2:
            shift = group_m - 1
            tile_n = in_group >> shift
            tile_m = first_m + (in_group & shift)
        else:
            tile_n = in_group // group_m
            tile_m = first_m + in_group % group_m
        return (tile_n, tile_m)

    return map_workgroup
