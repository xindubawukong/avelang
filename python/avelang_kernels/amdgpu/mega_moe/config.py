"""A MegaMoE solution plus launch geometry selected from the local token count."""

from dataclasses import dataclass
from typing import ClassVar

from .solutionid import DataType, MegaMoeSolutionId


@dataclass(frozen=True, slots=True)
class MegaMoeConfig:
    solution: MegaMoeSolutionId
    # Selected by get_2stage_cfgs; keep the geometry, not the raw token count,
    # in the configuration so equivalent launch buckets share cached kernels.
    stage1_tile_m: int = 32
    stage1_num_warps: int = 4
    max_tokens_per_rank: ClassVar[int] = 1024
    stage1_projection_n: ClassVar[int] = 256
    stage1_k_groups: ClassVar[int] = 1
    stage1_weight_load_aux: ClassVar[int] = 0

    def __post_init__(self):
        if not isinstance(self.solution, MegaMoeSolutionId):
            raise TypeError("solution must be a MegaMoeSolutionId")

    @property
    def compute_hidden(self):
        return (self.solution.hidden + 511) // 512 * 512

    @property
    def intermediate(self):
        return self.solution.intermediate

    @property
    def activation(self):
        return self.solution.activation

    @property
    def bias(self):
        return self.solution.bias_dtype == DataType.BF16

    @property
    def stage1_wave_m(self):
        return 64 if self.stage1_num_warps == 8 else 32

    @property
    def stage1_warps_n(self):
        return self.stage1_num_warps // (self.stage1_tile_m // self.stage1_wave_m)

    @property
    def stage1_wave_n(self):
        return self.stage1_projection_n // self.stage1_warps_n

    @property
    def stage1_input_stage_words(self):
        bm = self.stage1_tile_m
        return bm * 32 + (self.compute_hidden // 256) * (bm // 32) * 32

    @property
    def stage1_lds_words(self):
        return max(2 * self.stage1_input_stage_words, self.stage1_tile_m * self.stage1_projection_n) + 4

    @property
    def local_experts(self):
        return self.solution.experts // self.solution.world_size

    @property
    def input_token_bytes(self):
        d = self.solution.hidden
        return d // 2 + (d // 32 + 15) // 16 * 16

    @classmethod
    def validate_tokens(cls, tokens):
        if type(tokens) is not int or not 0 <= tokens <= cls.max_tokens_per_rank:
            raise ValueError("token count must be an integer in [0, 1024]")
