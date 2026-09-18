"""A MoE solution and its non-encoded problem parameters."""

from dataclasses import dataclass
from typing import ClassVar

from .solutionid import ActivationFunction, DataType, MoeSolutionId, WeightLoadPolicy


@dataclass(frozen=True, slots=True)
class MoeConfig:
    solution: MoeSolutionId
    experts: int
    topk: int

    # Fixed launch geometry for the current local implementations.
    stage2_workers: ClassVar[int] = 256

    stage1_num_warps: ClassVar[int] = 4
    stage2_num_warps: ClassVar[int] = 4

    def __post_init__(self):
        if not isinstance(self.solution, MoeSolutionId):
            raise TypeError("solution must be a MoeSolutionId; use from_solution for integer IDs")
        if type(self.experts) is not int or type(self.topk) is not int:
            raise TypeError("experts and topk must be integers")
        if self.experts <= 0 or self.topk <= 0 or self.topk > min(self.experts, 255):
            raise ValueError("experts and topk must be positive; topk must not exceed experts or 255")

    @classmethod
    def from_solution(cls, solution: MoeSolutionId | int, *, experts: int, topk: int) -> "MoeConfig":
        if not isinstance(solution, MoeSolutionId):
            solution = MoeSolutionId.from_int(solution)
        return cls(solution, experts, topk)

    @property
    def hidden(self) -> int:
        return self.solution.hidden

    @property
    def compute_hidden(self) -> int:
        return self.hidden

    @property
    def intermediate(self) -> int:
        return self.solution.intermediate

    @property
    def activation(self) -> ActivationFunction:
        return self.solution.activation

    @property
    def bias(self) -> bool:
        return self.solution.bias_dtype != DataType.NONE

    @property
    def weight_load_aux(self) -> int:
        return 2 if self.solution.weight_load_policy == WeightLoadPolicy.NON_TEMPORAL else 0

    @property
    def stage1_tile_m(self) -> int:
        return self.solution.stage1_tile_m

    @property
    def stage1_projection_n(self) -> int:
        return self.solution.stage1_tile_n // 2

    @property
    def stage2_tile_n(self) -> int:
        return 256

    @property
    def stage1_wave_m(self) -> int:
        return 32

    @property
    def stage1_warps_m(self) -> int:
        return self.stage1_tile_m // self.stage1_wave_m

    @property
    def stage1_warps_n(self) -> int:
        return self.stage1_num_warps // self.stage1_warps_m

    @property
    def stage1_wave_n(self) -> int:
        return self.stage1_projection_n // self.stage1_warps_n

    @property
    def stage1_input_stage_words(self) -> int:
        return self.stage1_tile_m * 32 + self.stage1_tile_m // 32 * 64

    @property
    def stage1_arena_words(self) -> int:
        return max(2 * self.stage1_input_stage_words, self.stage1_tile_m * self.stage1_projection_n)

    @property
    def stage1_lds_words(self) -> int:
        return self.stage1_arena_words

    @property
    def stage2_tile_m(self) -> int:
        return self.solution.stage2_tile_m

    @property
    def stage2_tile_k(self) -> int:
        return self.solution.stage2_tile_k

    @property
    def scale_columns(self) -> int:
        return self.intermediate // 32

    def stage1_grid(self, capacity: int):
        return ((self.intermediate // self.stage1_projection_n, capacity // self.stage1_tile_m, 1), (256, 1, 1))

    def stage2_grid(self):
        return ((self.hidden // 256, self.stage2_workers, 1), (256, 1, 1))
