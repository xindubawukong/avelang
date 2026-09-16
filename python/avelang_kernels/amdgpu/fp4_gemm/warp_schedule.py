"""Warp-level FP4 GEMM scheduling for AMDGPU."""

import avelang
import avelang.language as al

from .config import FP4GemmConfig
from .dequant import make_fp4_dequant
from .solution import MatmulMfmaType


def make_fp4_warp_schedule(
    config: FP4GemmConfig,
    shm_u32: int,
    _read_shm_a,
    _read_shm_b,
):
    WARP_ATOM_K = config.warp_atom_k
    WARP_ATOM_N = config.warp_atom_n
    M_TILES = config.m_tiles
    N_TILES = config.n_tiles
    READ_BATCH_A = config.read_batch_a
    LAYOUT_ELEMENTS_K = config.layout_elements_k
    LAYOUT_ELEMENTS_N = config.layout_elements_n
    ACC_TILES_PER_ATOM = config.layout_n // config.tile
    QWORDS = config.quant_vec_size
    IS_FP16 = config.mfma_type == MatmulMfmaType.FP16
    _dequant_scales, _dequant = make_fp4_dequant(config)

    @avelang.jit
    def _prefetch(
        shm: al.Tensor((shm_u32,), al.u32),
        stage: al.u32,
        data_a: al.Tensor((M_TILES, READ_BATCH_A, 4), al.u32),
        qword: al.Tensor((1, 4), al.u32),
        packed_scale: al.Tensor((1, 1), al.u16),
        warp_m: al.u32,
        warp_n: al.u32,
        warp_k: al.u32,
        wtid: al.u32,
    ):
        tile_idx_k = warp_k * WARP_ATOM_K
        _read_shm_a(
            shm, stage, warp_m, tile_idx_k, wtid, data_a
        )
        _read_shm_b(
            shm, stage, qword, packed_scale, warp_n, tile_idx_k, 0, wtid
        )

    @avelang.jit
    def _matmul(
        data_a: al.Tensor((M_TILES, READ_BATCH_A, 4), al.u32),
        qword: al.Tensor((1, 4), al.u32),
        packed_scale: al.Tensor((1, 1), al.u16),
        data_b: al.Tensor((1, 4), al.u32),
        n_atom: al.u32,
        acc: al.Tensor((M_TILES, N_TILES, 4), al.f32),
    ):
        scale_f32 = al.make_local((1, 2), al.f32)
        _dequant_scales(packed_scale[0, 0], scale_f32)

        # Petit WarpAccumLayout / WarpMatmulRegALayout: qword's K part
        # selects A, while its N part selects the accumulator and scale.
        accum = al.view(
            acc, al.f32,
            al.make_layout(
                (M_TILES, WARP_ATOM_N, (LAYOUT_ELEMENTS_K, LAYOUT_ELEMENTS_N), 4),
                (N_TILES * 4, ACC_TILES_PER_ATOM * 4, (0, 4), 1),
            ),
        )
        reg_a = al.view(
            data_a, al.u32,
            al.make_layout(
                (M_TILES, (LAYOUT_ELEMENTS_K, LAYOUT_ELEMENTS_N), 4),
                (READ_BATCH_A * 4, (4, 0), 1),
            ),
        )

        for j in al.range(QWORDS):
            _dequant(qword[0, j], scale_f32[0, j // LAYOUT_ELEMENTS_K], data_b)
            frag_b = al.view(data_b[0], al.Tensor((2, 2, 1), al.u32))
            for tile_m in al.range(M_TILES):
                frag_a = al.view(reg_a[tile_m, j], al.Tensor((2, 2, 1), al.u32))
                if IS_FP16:
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
    def _pipeline_compute(
        shm: al.Tensor((shm_u32,), al.u32),
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
        for k_atom in al.range(WARP_ATOM_K):
            tile_idx_k = warp_k * WARP_ATOM_K + k_atom
            for n_atom in al.range(WARP_ATOM_N):
                _matmul(
                    data_a, qword, packed_scale, data_b, n_atom, acc,
                )
                if n_atom + 1 < WARP_ATOM_N:
                    _read_shm_b(
                        shm, stage, qword, packed_scale,
                        warp_n, tile_idx_k, n_atom + 1, wtid,
                    )
            if k_atom + 1 < WARP_ATOM_K:
                tile_idx_k = tile_idx_k + 1
                _read_shm_a(
                    shm, stage, warp_m, tile_idx_k, wtid,
                    data_a,
                )
                _read_shm_b(
                    shm, stage, qword, packed_scale,
                    warp_n, tile_idx_k, 0, wtid,
                )

    return _prefetch, _pipeline_compute
