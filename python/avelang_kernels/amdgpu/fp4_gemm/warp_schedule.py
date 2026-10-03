"""Warp-level FP4 GEMM scheduling for AMDGPU."""

from __future__ import annotations

import avelang
import avelang.language as al

from .dequant import dequant, dequant_scales
from .memory_ops import read_shm_a, read_shm_b


@avelang.jit
def prefetch(
    config: al.constexpr,
    shm: al.Tensor((config.shm_u32,), al.u32),
    stage: al.u32,
    data_a: al.Tensor((config.gemm.m_tiles, config.gemm.read_batch_a, 4), al.u32),
    qword: al.Tensor((1, 4), al.u32),
    packed_scale: al.Tensor((1, 1), al.u16),
    warp_m: al.u32,
    warp_n: al.u32,
    warp_k: al.u32,
    wtid: al.u32,
):
    tile_idx_k = warp_k * config.gemm.warp_atom_k
    read_shm_a(
        config, shm, stage, warp_m, tile_idx_k, wtid, data_a
    )
    read_shm_b(
        config, shm, stage, qword, packed_scale, warp_n, tile_idx_k, 0, wtid
    )


@avelang.jit
def matmul(
    config: al.constexpr,
    data_a: al.Tensor((config.gemm.m_tiles, config.gemm.read_batch_a, 4), al.u32),
    qword: al.Tensor((1, 4), al.u32),
    packed_scale: al.Tensor((1, 1), al.u16),
    data_b: al.Tensor((1, 4), al.u32),
    n_atom: al.u32,
    acc: al.Tensor((config.gemm.m_tiles, config.gemm.n_tiles, 4), al.f32),
):
    scale_f32 = al.make_local((1, 2), al.f32)
    dequant_scales(config, packed_scale[0, 0], scale_f32)

    # Petit WarpAccumLayout / WarpMatmulRegALayout: qword's K part
    # selects A, while its N part selects the accumulator and scale.
    accum = al.view(
        acc, al.f32,
        al.make_layout(
            (config.gemm.m_tiles, config.gemm.warp_atom_n, (config.gemm.layout_elements_k, config.gemm.layout_elements_n), 4),
            (config.gemm.n_tiles * 4, config.gemm.layout_n // config.gemm.tile * 4, (0, 4), 1),
        ),
    )
    reg_a = al.view(
        data_a, al.u32,
        al.make_layout(
            (config.gemm.m_tiles, (config.gemm.layout_elements_k, config.gemm.layout_elements_n), 4),
            (config.gemm.read_batch_a * 4, (4, 0), 1),
        ),
    )

    for j in al.range(config.gemm.quant_vec_size):
        dequant(config, qword[0, j], scale_f32[0, j // config.gemm.layout_elements_k], data_b)
        frag_b = al.view(data_b[0], al.Tensor((2, 2, 1), al.u32))
        for tile_m in al.range(config.gemm.m_tiles):
            frag_a = al.view(reg_a[tile_m, j], al.Tensor((2, 2, 1), al.u32))
            if config.gemm.is_fp16:
                accum[tile_m, n_atom, j] = al.amdgpu.mfma_16x16x16_f16_f32(
                    frag_b[0], frag_a[0], accum[tile_m, n_atom, j]
                )
                accum[tile_m, n_atom, j] = al.amdgpu.mfma_16x16x16_f16_f32(
                    frag_b[1], frag_a[1], accum[tile_m, n_atom, j]
                )
            else:
                accum[tile_m, n_atom, j] = al.amdgpu.mfma_16x16x16_bf16_f32(
                    frag_b[0], frag_a[0], accum[tile_m, n_atom, j]
                )
                accum[tile_m, n_atom, j] = al.amdgpu.mfma_16x16x16_bf16_f32(
                    frag_b[1], frag_a[1], accum[tile_m, n_atom, j]
                )


@avelang.jit
def pipeline_compute(
    config: al.constexpr,
    shm: al.Tensor((config.shm_u32,), al.u32),
    stage: al.u32,
    data_a: al.Tensor((config.gemm.m_tiles, config.gemm.read_batch_a, 4), al.u32),
    qword: al.Tensor((1, 4), al.u32),
    packed_scale: al.Tensor((1, 1), al.u16),
    data_b: al.Tensor((1, 4), al.u32),
    warp_m: al.u32,
    warp_n: al.u32,
    warp_k: al.u32,
    wtid: al.u32,
    acc: al.Tensor((config.gemm.m_tiles, config.gemm.n_tiles, 4), al.f32),
):
    for k_atom in al.range(config.gemm.warp_atom_k):
        tile_idx_k = warp_k * config.gemm.warp_atom_k + k_atom
        for n_atom in al.range(config.gemm.warp_atom_n):
            matmul(
                config, data_a, qword, packed_scale, data_b, n_atom, acc,
            )
            if n_atom + 1 < config.gemm.warp_atom_n:
                read_shm_b(
                    config, shm, stage, qword, packed_scale,
                    warp_n, tile_idx_k, n_atom + 1, wtid,
                )
        if k_atom + 1 < config.gemm.warp_atom_k:
            tile_idx_k = tile_idx_k + 1
            read_shm_a(
                config, shm, stage, warp_m, tile_idx_k, wtid,
                data_a,
            )
            read_shm_b(
                config, shm, stage, qword, packed_scale,
                warp_n, tile_idx_k, 0, wtid,
            )
