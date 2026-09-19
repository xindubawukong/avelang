"""Petit MegaMoE's compact ID, with topology encoded in bits 24 through 51."""

from dataclasses import dataclass
from enum import IntEnum

from ..local_moe.solutionid import (
    BASE_FIELDS,
    ActivationFunction,
    DataType,
    MfmaShape,
    Stage1Buffering,
    Stages,
    WeightOrdering,
)


class ProducerGeometry(IntEnum):
    CTA56 = 0
    CTA64 = 1
    CTA128 = 2
    CTA192 = 3

    @property
    def blocks(self):
        return (56, 64, 128, 192)[int(self)]


class W2TileShape(IntEnum):
    N256 = 0
    N128 = 1


@dataclass(frozen=True, slots=True)
class MegaMoeSolutionId:
    world_size: int
    experts: int
    topk: int
    hidden: int
    intermediate: int
    activation: ActivationFunction
    bias_dtype: DataType
    act_dtype: DataType = DataType.MXFP4
    weight_dtype: DataType = DataType.MXFP4
    weight_ordering: WeightOrdering = WeightOrdering.NATIVE_MXFP4
    mfma: MfmaShape = MfmaShape.SCALE_FP4_MXFP4
    stages: Stages = Stages.TWO_STAGE
    stage1_buffering: Stage1Buffering = Stage1Buffering.DOUBLE_BUFFER
    producer_geometry: ProducerGeometry = ProducerGeometry.CTA56
    w2_tile_shape: W2TileShape = W2TileShape.N256

    def __post_init__(self):
        for name in ("world_size", "experts", "topk", "hidden", "intermediate"):
            if type(getattr(self, name)) is not int:
                raise TypeError(f"{name} must be an integer")
        for name, enum in [(n, e) for n, e, _, _ in BASE_FIELDS] + [
            ("producer_geometry", ProducerGeometry),
            ("w2_tile_shape", W2TileShape),
        ]:
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be a {enum.__name__} or integer")
            object.__setattr__(self, name, enum(value))
        if self.world_size not in (2, 4, 8):
            raise ValueError("world_size must be 2, 4 or 8")
        if not 32 <= self.experts <= 1024 or self.experts % 32 or self.experts % self.world_size:
            raise ValueError("experts must be a multiple of 32 in [32, 1024], divisible by world_size")
        if not 1 <= self.topk <= min(31, self.experts):
            raise ValueError("topk must be in [1, min(31, experts)]")
        if not 64 <= self.hidden <= 16320 or self.hidden % 64:
            raise ValueError("hidden must be a positive multiple of 64, at most 16320")
        if not 512 <= self.intermediate <= 16384 or self.intermediate % 512:
            raise ValueError("intermediate must be a positive multiple of 512, at most 16384")

    def __int__(self):
        return (
            sum(int(getattr(self, name)) << shift for name, _, shift, _ in BASE_FIELDS)
            | ((self.world_size.bit_length() - 1) << 24)
            | (((self.experts // 32 - 1) & 15) << 26)
            | (((self.experts // 32 - 1) >> 4) << 50)
            | ((self.topk & 15) << 30)
            | ((self.topk >> 4) << 51)
            | ((self.hidden // 64) << 34)
            | (int(self.w2_tile_shape) << 42)
            | ((self.intermediate // 512 - 1) << 43)
            | (int(self.producer_geometry) << 48)
        )

    @classmethod
    def from_int(cls, value):
        if type(value) is not int:
            raise TypeError("solution ID must be an integer")
        if value < 0 or value >> 52:
            raise ValueError("MegaMoE solution ID must have reserved bits [52,63] clear")
        fields = {name: enum((value >> shift) & ((1 << bits) - 1)) for name, enum, shift, bits in BASE_FIELDS}
        return cls(
            world_size=1 << ((value >> 24) & 3),
            experts=(((value >> 26) & 15) + ((value >> 50) & 1) * 16 + 1) * 32,
            topk=((value >> 30) & 15) | (((value >> 51) & 1) << 4),
            hidden=((value >> 34) & 255) * 64,
            intermediate=(((value >> 43) & 31) + 1) * 512,
            w2_tile_shape=W2TileShape((value >> 42) & 1),
            producer_geometry=ProducerGeometry((value >> 48) & 3),
            **fields,
        )
