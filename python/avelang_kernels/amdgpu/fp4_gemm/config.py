"""Compile-time configuration shared by FP4 GEMM components."""

from dataclasses import dataclass

from .solution import (
    LAYOUT_K,
    LAYOUT_N,
    TILE,
    MatmulElementB,
    MatmulFeatures,
    MatmulMfmaType,
    MatmulWarpPartition,
    SolutionId,
    fp4_group_size,
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
    solution: SolutionId
    arch: str = "gfx942"

    @classmethod
    def from_solution(cls, solution: SolutionId, arch: str = "gfx942") -> "FP4GemmConfig":
        _ = solution.shape
        return cls(solution, arch)

    @property
    def use_bf8(self) -> bool:
        return self.mfma_type == MatmulMfmaType.BF16 and self.arch in ("gfx942", "gfx950")

    @property
    def dequant_bias(self) -> float:
        # CDNA3 BF8 uses bias 16; OCP BF8 and the FP16 path use bias 15.
        return 32768.0 if self.use_bf8 and self.arch == "gfx942" else 16384.0

    @property
    def global_scale_factor(self) -> float:
        if self.high_precision:
            return 1.0
        return self.dequant_bias / (128.0 if self.element_b == MatmulElementB.NVFP4 else 1.0)

    @property
    def features(self) -> MatmulFeatures:
        return self.solution.features

    @property
    def element_b(self) -> MatmulElementB:
        return self.solution.element_b

    @property
    def mfma_type(self) -> MatmulMfmaType:
        return self.solution.mfma_type

    @property
    def high_precision(self) -> bool:
        return bool(self.features & MatmulFeatures.HIGH_PRECISION)

    @property
    def warp_partition(self) -> MatmulWarpPartition:
        return self.solution.warp_partition

    @property
    def tile(self) -> int:
        return TILE

    @property
    def layout_k(self) -> int:
        return LAYOUT_K

    @property
    def layout_n(self) -> int:
        return LAYOUT_N

    @property
    def layout_elements_k(self) -> int:
        return 2

    @property
    def layout_elements_n(self) -> int:
        return 2

    @property
    def pipeline_stages(self) -> int:
        return PIPELINE_STAGES

    @property
    def warp_size(self) -> int:
        return WARP_SIZE

    @property
    def max_shm_bytes(self) -> int:
        return MAX_SHM_BYTES

    @property
    def element_a_bytes(self) -> int:
        if self.mfma_type in (MatmulMfmaType.FP16, MatmulMfmaType.BF16):
            return 2
        if self.mfma_type == MatmulMfmaType.FP8:
            return 1
        raise ValueError(f"unsupported MFMA type {self.mfma_type}")

    @property
    def output_bytes(self) -> int:
        return 2

    @property
    def vec_size(self) -> int:
        return UINT4_BYTES // self.element_a_bytes

    @property
    def scale_vec_size(self) -> int:
        return SCALE_VEC_SIZE

    @property
    def pack_factor(self) -> int:
        return PACK_FACTOR

    @property
    def quant_vec_size(self) -> int:
        return QUANT_VEC_SIZE

    @property
    def packed_values_per_vector(self) -> int:
        return self.pack_factor * self.quant_vec_size

    @property
    def read_batch_a(self) -> int:
        return (
            self.layout_elements_k
            * self.pack_factor
            * self.element_a_bytes
            // UINT4_BYTES
        )

    @property
    def result_vec_size(self) -> int:
        return UINT4_BYTES // self.output_bytes

    @property
    def shm_vec_size(self) -> int:
        return (2 * U32_BYTES) // self.output_bytes

    @property
    def use_zero_points(self) -> bool:
        return False

    @property
    def zero_points_in_shm(self) -> bool:
        return False

    @property
    def num_tiles_m(self) -> int:
        return self.solution.tile_m

    @property
    def num_tiles_n(self) -> int:
        return self.solution.tile_n

    @property
    def num_tiles_k(self) -> int:
        return self.solution.tile_k * 4

    @property
    def group_m(self) -> int:
        return self.num_tiles_m * self.tile

    @property
    def group_n(self) -> int:
        return self.num_tiles_n * self.tile

    @property
    def group_k(self) -> int:
        return self.num_tiles_k * self.tile

    @property
    def group_size(self) -> int:
        return fp4_group_size(self.element_b)

    @property
    def warp_m(self) -> int:
        return self.solution.warp_partition_m

    @property
    def warp_n(self) -> int:
        return self.solution.warp_partition_n

    @property
    def warp_k(self) -> int:
        return self.solution.warp_partition_k

    @property
    def num_warps(self) -> int:
        return self.warp_m * self.warp_n * self.warp_k

    @property
    def threads(self) -> int:
        return self.num_warps * self.warp_size

    @property
    def warp_tile_m(self) -> int:
        return self.group_m // self.warp_m

    @property
    def warp_tile_n(self) -> int:
        return self.group_n // self.warp_n

    @property
    def warp_atom_k(self) -> int:
        return self.group_k // self.layout_k // self.warp_k

    @property
    def warp_atom_n(self) -> int:
        return self.group_n // self.layout_n // self.warp_n

    @property
    def m_tiles(self) -> int:
        return self.warp_tile_m // self.tile

    @property
    def n_tiles(self) -> int:
        return self.warp_tile_n // self.tile

    @property
    def accumulator_shape(self) -> tuple[int, int, int]:
        return (self.m_tiles, self.n_tiles, 4)
