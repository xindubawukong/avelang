"""Petit's 64-bit local MoE solution encoding (not the MegaMoE encoding)."""

from dataclasses import dataclass
from enum import IntEnum


class DataType(IntEnum):
    NONE = 0
    MXFP4 = 1
    NVFP4 = 2
    CHANNEL_SCALE_FP8 = 3
    BLOCK_SCALE_FP8 = 4
    BF16 = 5


class WeightOrdering(IntEnum):
    NATIVE_MXFP4 = 0
    PETIT_MXFP4 = 1
    PETIT_FP8 = 2


class MfmaShape(IntEnum):
    FP8_16X16X32 = 0
    BF16_MXFP4 = 1
    SCALE_FP4_MXFP4 = 2


class Stages(IntEnum):
    ONE_STAGE = 0
    TWO_STAGE = 1


class ActivationFunction(IntEnum):
    SILU_DOT = 0
    OPENAI_SWIGLU = 1
    SITU_V2 = 2


class Stage1Buffering(IntEnum):
    SINGLE_BUFFER = 0
    DOUBLE_BUFFER = 1


class WeightLoadPolicy(IntEnum):
    CACHED = 0
    NON_TEMPORAL = 1


class Stage1TileShape(IntEnum):
    # N includes both gate and up projections: M32 has 128 columns each.
    M32_N256 = 0
    M64_N512 = 1
    M64_N256 = 2
    M32_N128_K2 = 4  # Value 3 is reserved in Petit.


class Stage2TileShape(IntEnum):
    M32_N256_K256 = 0
    M32_N256_K128 = 1
    M64_N256_K128 = 2


# Field positions match FusedMoESolutionId::Repr/FromRepr in fused_moe.h.
# MegaMoE reuses these low 24 bits from the local implementation.
BASE_FIELDS = (
    ("act_dtype", DataType, 0, 4),
    ("weight_dtype", DataType, 4, 4),
    ("bias_dtype", DataType, 8, 4),
    ("weight_ordering", WeightOrdering, 12, 2),
    ("mfma", MfmaShape, 14, 2),
    ("stages", Stages, 16, 4),
    ("activation", ActivationFunction, 20, 3),
    ("stage1_buffering", Stage1Buffering, 23, 1),
)
_FIELDS = BASE_FIELDS + (
    ("stage1_weight_load_policy", WeightLoadPolicy, 40, 1),
    ("stage2_weight_load_policy", WeightLoadPolicy, 41, 1),
)
_SHAPES = (("hidden", 24), ("intermediate", 32))
# The local tile fields occupy interleaved bits in the current Petit ABI.
_TILES = (("stage1_tile_shape", Stage1TileShape, (42, 44, 46)), ("stage2_tile_shape", Stage2TileShape, (43, 45)))


@dataclass(frozen=True, slots=True)
class MoeSolutionId:
    """A complete local solution ID with encoded D/I dimensions.

    Encoding validates the bit layout and enum values. Dispatch separately
    resolves complete solutions to the available kernel implementations.
    Token count, expert count, top-k and launch worker count are not encoded.
    """

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
    stage1_weight_load_policy: WeightLoadPolicy = WeightLoadPolicy.CACHED
    stage2_weight_load_policy: WeightLoadPolicy = WeightLoadPolicy.CACHED
    stage1_tile_shape: Stage1TileShape = Stage1TileShape.M32_N256
    stage2_tile_shape: Stage2TileShape = Stage2TileShape.M32_N256_K256

    def __post_init__(self):
        enum_fields = [(name, enum) for name, enum, _, _ in _FIELDS] + [(name, enum) for name, enum, _ in _TILES]
        for name, enum in enum_fields:
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be a {enum.__name__} or integer")
            object.__setattr__(self, name, enum(value))
        for name, _ in _SHAPES:
            value = getattr(self, name)
            if type(value) is not int:
                raise TypeError(f"{name} must be an integer")
            if value <= 0 or value % 64 or value // 64 > 255:
                raise ValueError(f"{name} must be a positive multiple of 64, at most 16320")

    @classmethod
    def from_int(cls, value: int) -> "MoeSolutionId":
        if type(value) is not int:
            raise TypeError("solution ID must be an integer")
        if value < 0 or value >> 47:
            raise ValueError("local MoE solution ID must be nonnegative with reserved bits [47,63] clear")
        fields = {name: enum((value >> shift) & ((1 << bits) - 1)) for name, enum, shift, bits in _FIELDS}
        fields.update({name: ((value >> shift) & 255) * 64 for name, shift in _SHAPES})
        fields.update(
            {
                name: enum(sum(((value >> bit) & 1) << index for index, bit in enumerate(bits)))
                for name, enum, bits in _TILES
            }
        )
        return cls(**fields)

    def __int__(self) -> int:
        fields = sum(int(getattr(self, name)) << shift for name, _, shift, _ in _FIELDS)
        shapes = sum((getattr(self, name) // 64) << shift for name, shift in _SHAPES)
        tiles = sum(
            ((int(getattr(self, name)) >> index) & 1) << bit
            for name, _, bits in _TILES
            for index, bit in enumerate(bits)
        )
        return fields | shapes | tiles

    @property
    def stage1_tile_m(self) -> int:
        return 64 if self.stage1_tile_shape in (Stage1TileShape.M64_N512, Stage1TileShape.M64_N256) else 32

    @property
    def stage1_tile_n(self) -> int:
        """Combined gate/up width, matching Petit's naming."""
        if self.stage1_tile_shape == Stage1TileShape.M32_N128_K2:
            return 128
        return 512 if self.stage1_tile_shape == Stage1TileShape.M64_N512 else 256

    @property
    def stage1_k_groups(self) -> int:
        return 2 if self.stage1_tile_shape == Stage1TileShape.M32_N128_K2 else 1

    @property
    def stage2_tile_m(self) -> int:
        return 64 if self.stage2_tile_shape == Stage2TileShape.M64_N256_K128 else 32

    @property
    def stage2_tile_k(self) -> int:
        return 256 if self.stage2_tile_shape == Stage2TileShape.M32_N256_K256 else 128
