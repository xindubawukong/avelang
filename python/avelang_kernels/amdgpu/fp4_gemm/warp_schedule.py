"""Warp-level FP4 GEMM scheduling for AMDGPU."""

from __future__ import annotations

import avelang
import avelang.language as al

from .config import (
    LAYOUT_ELEMENTS_K,
    LAYOUT_ELEMENTS_N,
    LAYOUT_N,
    QUANT_VEC_SIZE,
    TILE,
)

from .dequant import make_dequant
from .solution import MatmulMfmaType


def make_warp_schedule(config, memory, read_shm_a, read_shm_b):
    dequant_scales, dequant = make_dequant(config)
    M_TILES = al.constexpr(config.m_tiles)
    N_TILES = al.constexpr(config.n_tiles)
    READ_BATCH_A = al.constexpr(config.read_batch_a)
    WARP_ATOM_K = al.constexpr(config.warp_atom_k)
    WARP_ATOM_N = al.constexpr(config.warp_atom_n)
    MFMA_TYPE = al.constexpr(config.mfma_type)
    FP16 = MatmulMfmaType.FP16
    SHM_U32 = al.constexpr(memory.shm_u32)

    @avelang.jit
    def prefetch(
        shm: al.Tensor((SHM_U32,), al.u32),
        stage: al.u32,
        data_a: al.Tensor((M_TILES, READ_BATCH_A, 4), al.u32),
        qword: al.Tensor((1, 4), al.u32),
        packed_scale: al.Tensor((1, 1), al.u16),
        warp_m: al.u32,
        warp_n: al.u32,
        warp_k: al.u32,
        wtid: al.u32,
    ):
        # Keep annotation-only constants in the Python closure.
        _ = M_TILES
        _ = READ_BATCH_A
        _ = SHM_U32
        tile_idx_k = warp_k * WARP_ATOM_K
        read_shm_a(
            shm, stage, warp_m, tile_idx_k, wtid, data_a
        )
        read_shm_b(
            shm, stage, qword, packed_scale, warp_n, tile_idx_k, 0, wtid
        )

    @avelang.jit
    def matmul(
        data_a: al.Tensor((M_TILES, READ_BATCH_A, 4), al.u32),
        qword: al.Tensor((1, 4), al.u32),
        packed_scale: al.Tensor((1, 1), al.u16),
        data_b: al.Tensor((1, 4), al.u32),
        n_atom: al.u32,
        acc: al.Tensor((M_TILES, N_TILES, 4), al.f32),
    ):
        scale_f32 = al.make_local((1, 2), al.f32)
        dequant_scales(packed_scale[0, 0], scale_f32)

        # Petit WarpAccumLayout / WarpMatmulRegALayout: qword's K part
        # selects A, while its N part selects the accumulator and scale.
        accum = al.view(
            acc, al.f32,
            al.make_layout(
                (M_TILES, WARP_ATOM_N, (LAYOUT_ELEMENTS_K, LAYOUT_ELEMENTS_N), 4),
                (N_TILES * 4, LAYOUT_N // TILE * 4, (0, 4), 1),
            ),
        )
        reg_a = al.view(
            data_a, al.u32,
            al.make_layout(
                (M_TILES, (LAYOUT_ELEMENTS_K, LAYOUT_ELEMENTS_N), 4),
                (READ_BATCH_A * 4, (4, 0), 1),
            ),
        )

        for j in al.range(QUANT_VEC_SIZE):
            dequant(qword[0, j], scale_f32[0, j // LAYOUT_ELEMENTS_K], data_b)
            frag_b = al.view(data_b[0], al.Tensor((2, 2, 1), al.u32))
            for tile_m in al.range(M_TILES):
                frag_a = al.view(reg_a[tile_m, j], al.Tensor((2, 2, 1), al.u32))
                if MFMA_TYPE == FP16:
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
        shm: al.Tensor((SHM_U32,), al.u32),
        stage: al.u32,
        data_a: al.Tensor((M_TILES, READ_BATCH_A, 4), al.u32),
        qword: al.Tensor((1, 4), al.u32),
        packed_scale: al.Tensor((1, 1), al.u16),
        data_b: al.Tensor((1, 4), al.u32),
        warp_m: al.u32,
        warp_n: al.u32,
        warp_k: al.u32,
        wtid: al.u32,
        acc: al.Tensor((M_TILES, N_TILES, 4), al.f32),
    ):
        _ = M_TILES
        _ = N_TILES
        _ = READ_BATCH_A
        _ = SHM_U32
        for k_atom in al.range(WARP_ATOM_K):
            tile_idx_k = warp_k * WARP_ATOM_K + k_atom
            for n_atom in al.range(WARP_ATOM_N):
                matmul(
                    data_a, qword, packed_scale, data_b, n_atom, acc,
                )
                if n_atom + 1 < WARP_ATOM_N:
                    read_shm_b(
                        shm, stage, qword, packed_scale,
                        warp_n, tile_idx_k, n_atom + 1, wtid,
                    )
            if k_atom + 1 < WARP_ATOM_K:
                tile_idx_k = tile_idx_k + 1
                read_shm_a(
                    shm, stage, warp_m, tile_idx_k, wtid,
                    data_a,
                )
                read_shm_b(
                    shm, stage, qword, packed_scale,
                    warp_n, tile_idx_k, 0, wtid,
                )

    return prefetch, pipeline_compute
