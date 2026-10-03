"""Compile-time configuration shared by FP4 GEMM components."""

from dataclasses import dataclass

from .solution import (
    LAYOUT_K,
    LAYOUT_N,
    TILE,
    MatmulElementB,
    MatmulFeatures,
    MatmulMfmaType,
    SolutionId,
    fp4_scale_group_size,
)

WARP_SIZE = 64
PIPELINE_STAGES = 2
MAX_SHM_BYTES = 64 * 1024
UINT4_BYTES = 16
U32_BYTES = 4
U16_BYTES = 2
FP4_BITS = 4
PACK_FACTOR = 32 // FP4_BITS
QUANT_VEC_SIZE = UINT4_BYTES // U32_BYTES
SCALE_VEC_SIZE = UINT4_BYTES


@dataclass(frozen=True, slots=True)
class FP4GemmConfig:
    use_bf8: bool
    global_scale_factor: float
    high_precision: bool
    element_a_bytes: int
    vec_size: int
    read_batch_a: int
    group_m: int
    group_n: int
    group_k: int
    scale_group_size: int
    warp_m: int
    warp_n: int
    warp_k: int
    num_warps: int
    threads: int
    warp_tile_m: int
    warp_tile_n: int
    warp_atom_k: int
    warp_atom_n: int
    m_tiles: int
    n_tiles: int
    is_fp16: bool
    is_mxfp4: bool
    weight_bias: float
    tile: int = TILE
    layout_k: int = LAYOUT_K
    layout_n: int = LAYOUT_N
    layout_elements_k: int = 2
    layout_elements_n: int = 2
    pipeline_stages: int = PIPELINE_STAGES
    warp_size: int = WARP_SIZE
    max_shm_bytes: int = MAX_SHM_BYTES
    output_bytes: int = U16_BYTES
    scale_vec_size: int = SCALE_VEC_SIZE
    pack_factor: int = PACK_FACTOR
    quant_vec_size: int = QUANT_VEC_SIZE
    packed_values_per_vector: int = PACK_FACTOR * QUANT_VEC_SIZE
    result_vec_size: int = UINT4_BYTES // U16_BYTES
    shm_vec_size: int = 2 * U32_BYTES // U16_BYTES
    accum_values: int = 4

    @classmethod
    def from_solution(cls, solution: SolutionId, arch: str = "gfx942") -> "FP4GemmConfig":
        use_bf8 = solution.mfma_type == MatmulMfmaType.BF16 and arch in ("gfx942", "gfx950")
        dequant_bias = 32768.0 if use_bf8 and arch == "gfx942" else 16384.0
        high_precision = bool(solution.features & MatmulFeatures.HIGH_PRECISION)
        is_mxfp4 = solution.element_b == MatmulElementB.MXFP4
        if high_precision:
            global_scale_factor = 1.0
        elif is_mxfp4:
            global_scale_factor = 16384.0 if arch in ("gfx950", "gfx1200", "gfx1201") else 32768.0
        else:
            global_scale_factor = dequant_bias / 128.0
        element_a_bytes = 2 if solution.mfma_type in (MatmulMfmaType.FP16, MatmulMfmaType.BF16) else 1
        warp_tile_m = solution.group_m // solution.warp_partition_m
        warp_tile_n = solution.group_n // solution.warp_partition_n
        return cls(
            use_bf8=use_bf8,
            global_scale_factor=global_scale_factor,
            high_precision=high_precision,
            element_a_bytes=element_a_bytes,
            vec_size=UINT4_BYTES // element_a_bytes,
            read_batch_a=2 * PACK_FACTOR * element_a_bytes // UINT4_BYTES,
            group_m=solution.group_m,
            group_n=solution.group_n,
            group_k=solution.group_k,
            scale_group_size=fp4_scale_group_size(solution.element_b),
            warp_m=solution.warp_partition_m,
            warp_n=solution.warp_partition_n,
            warp_k=solution.warp_partition_k,
            num_warps=solution.num_warps,
            threads=solution.num_warps * WARP_SIZE,
            warp_tile_m=warp_tile_m,
            warp_tile_n=warp_tile_n,
            warp_atom_k=solution.group_k // LAYOUT_K // solution.warp_partition_k,
            warp_atom_n=solution.group_n // LAYOUT_N // solution.warp_partition_n,
            m_tiles=warp_tile_m // TILE,
            n_tiles=warp_tile_n // TILE,
            is_fp16=solution.mfma_type == MatmulMfmaType.FP16,
            is_mxfp4=is_mxfp4,
            weight_bias=dequant_bias / 128.0 if high_precision else 1.0,
        )

    @property
    def accumulator_shape(self) -> tuple[int, int, int]:
        return (self.m_tiles, self.n_tiles, self.accum_values)
