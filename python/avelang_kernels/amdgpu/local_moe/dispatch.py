"""Problem-to-configuration selection for dynamic MXFP4 two-stage local MoE."""

from enum import Enum
from functools import lru_cache

import torch

from .config import MoeConfig
from .solutionid import ActivationFunction, DataType, MoeSolutionId, Stage1TileShape, Stage2TileShape, WeightLoadPolicy


def _normalize_activation(value) -> ActivationFunction:
    if isinstance(value, ActivationFunction):
        return value
    if type(value) is int:
        return ActivationFunction(value)
    if isinstance(value, Enum):
        value = value.name
    if isinstance(value, str):
        name = value.lower()
        if name in ("silu", "silu_dot"):
            return ActivationFunction.SILU_DOT
        if name in ("swiglu", "openai_swiglu"):
            return ActivationFunction.OPENAI_SWIGLU
        if name in ("situ", "situ_v2", "situv2"):
            return ActivationFunction.SITU_V2
    raise ValueError(f"unsupported activation: {value!r}")


def _normalize_data_type(value) -> DataType:
    if value is None:
        return DataType.NONE
    if isinstance(value, DataType):
        return value
    if type(value) is int:
        return DataType(value)
    if isinstance(value, Enum):
        value = value.name
    elif isinstance(value, torch.dtype):
        value = str(value)
    if isinstance(value, str):
        name = value.removeprefix("torch.").lower()
        if name == "bfloat16":
            return DataType.BF16
        if name in ("uint8", "fp4x2", "float4_e2m1fn_x2"):
            return DataType.MXFP4
        try:
            return DataType[name.upper()]
        except KeyError:
            pass
    raise ValueError(f"unsupported data type: {value!r}")


def _normalize_weight_load_policy(value) -> WeightLoadPolicy | None:
    if value is None:
        return None
    if isinstance(value, WeightLoadPolicy):
        return value
    if type(value) is int:
        return WeightLoadPolicy(value)
    if isinstance(value, Enum):
        value = value.name
    if isinstance(value, str):
        name = value.lower()
        if name == "auto":
            return None
        if name == "cached":
            return WeightLoadPolicy.CACHED
        if name == "non_temporal":
            return WeightLoadPolicy.NON_TEMPORAL
    raise ValueError(f"unsupported weight load policy: {value!r}")


def _normalize_request(model_dim, inter_dim, activation, bias_dtype, dtype, q_dtype_a, q_dtype_w, group_size, arch):
    if not isinstance(arch, str):
        raise TypeError("arch must be a string")
    arch = arch.split(":", 1)[0]
    if arch != "gfx950":
        raise ValueError(f"dynamic MXFP4 two-stage MoE requires gfx950, got {arch}")
    if _normalize_data_type(dtype) != DataType.BF16:
        raise ValueError("dynamic MXFP4 MoE requires BF16 input/output")
    if type(group_size) is not int or group_size != 32:
        raise ValueError("dynamic MXFP4 MoE requires group_size=32")
    solution = MoeSolutionId(
        hidden=model_dim,
        intermediate=inter_dim,
        activation=_normalize_activation(activation),
        bias_dtype=_normalize_data_type(bias_dtype),
        act_dtype=_normalize_data_type(q_dtype_a),
        weight_dtype=_normalize_data_type(q_dtype_w),
    )
    return (solution, arch)


_OPERATION_FIELDS = (
    "hidden",
    "intermediate",
    "act_dtype",
    "weight_dtype",
    "bias_dtype",
    "activation",
    "weight_ordering",
)


@lru_cache(maxsize=2048)
def _registered_2stage_implementations(hidden, intermediate):
    from .stage1 import make_stage1_kernel
    from .stage2 import make_stage2_kernel

    implementations = {}
    shapes = (
        (Stage1TileShape.M32_N256, Stage2TileShape.M32_N256_K256),
        (Stage1TileShape.M64_N512, Stage2TileShape.M32_N256_K256),
        (Stage1TileShape.M32_N256, Stage2TileShape.M32_N256_K128),
    )
    for activation in ActivationFunction:
        for bias in (DataType.NONE, DataType.BF16):
            for s1_shape, s2_shape in shapes:
                if s2_shape != Stage2TileShape.M32_N256_K256 and (
                    bias != DataType.NONE or activation != ActivationFunction.SITU_V2
                ):
                    continue
                for p1 in WeightLoadPolicy:
                    for p2 in WeightLoadPolicy:
                        solution = MoeSolutionId(
                            hidden=hidden,
                            intermediate=intermediate,
                            activation=activation,
                            bias_dtype=bias,
                            stage1_tile_shape=s1_shape,
                            stage2_tile_shape=s2_shape,
                            stage1_weight_load_policy=p1,
                            stage2_weight_load_policy=p2,
                        )
                        if (
                            hidden % (256 * 1)
                            or intermediate % (solution.stage1_tile_n // 2)
                            or intermediate % solution.stage2_tile_k
                        ):
                            continue
                        implementations[solution] = (make_stage1_kernel, make_stage2_kernel)
    return implementations


def resolve_2stage_implementation(config: MoeConfig):
    """Resolve a complete solution to its registered pair of kernel factories."""
    implementations = _registered_2stage_implementations(config.hidden, config.intermediate)
    try:
        return implementations[config.solution]
    except KeyError:
        raise ValueError(f"unsupported Ave local MoE solution: {int(config.solution):#x}") from None


def _matching_solutions(config):
    implementations = _registered_2stage_implementations(config.hidden, config.intermediate)
    candidates = tuple(
        solution
        for solution in implementations
        if all(getattr(solution, field) == getattr(config.solution, field) for field in _OPERATION_FIELDS)
    )
    if not candidates:
        raise ValueError(f"unsupported Ave local MoE solution: {int(config.solution):#x}")
    return candidates


def available_2stage_solutions(
    model_dim: int,
    inter_dim: int,
    expert: int,
    topk: int,
    *,
    activation: ActivationFunction | str,
    bias_dtype: DataType | str,
    dtype: DataType | str = DataType.BF16,
    q_dtype_a: DataType | str = DataType.MXFP4,
    q_dtype_w: DataType | str = DataType.MXFP4,
    group_size: int = 32,
    arch: str = "gfx950",
) -> tuple[MoeSolutionId, ...]:
    """Return legal cached/NT candidates; reject unsupported problem formats."""
    solution, _ = _normalize_request(
        model_dim, inter_dim, activation, bias_dtype, dtype, q_dtype_a, q_dtype_w, group_size, arch
    )
    return _matching_solutions(MoeConfig(solution, expert, topk))


@lru_cache(maxsize=2048)
def _get_2stage_cfgs_cached(token, requested, arch, policy1, policy2, explicit):
    kimi = (requested.hidden, requested.intermediate, requested.topk, requested.activation) == (
        3584,
        384,
        16,
        ActivationFunction.SITU_V2,
    )
    if explicit is not None:
        config = MoeConfig(explicit, requested.experts, requested.topk)
        resolve_2stage_implementation(config)
        for field in _OPERATION_FIELDS:
            if getattr(explicit, field) != getattr(requested.solution, field):
                raise ValueError(f"solution_id does not match requested {field}")
        for stage, policy in ((1, policy1), (2, policy2)):
            if policy is not None and getattr(explicit, f"stage{stage}_weight_load_policy") != policy:
                raise ValueError(f"solution_id does not match requested stage{stage}_weight_load_policy")
        selected = explicit
    else:
        candidates = _matching_solutions(requested)
        if kimi:
            preferred1 = preferred2 = (
                WeightLoadPolicy.NON_TEMPORAL
                if token * requested.topk // requested.experts < 64
                else WeightLoadPolicy.CACHED
            )
            s1_shape = Stage1TileShape.M32_N256
            s2_shape = Stage2TileShape.M32_N256_K128
        else:
            preferred1 = preferred2 = (
                WeightLoadPolicy.NON_TEMPORAL
                if token * requested.topk // requested.experts < 64
                else WeightLoadPolicy.CACHED
            )
            s1_shape, s2_shape = (Stage1TileShape.M32_N256, Stage2TileShape.M32_N256_K256)
        matches = tuple(
            s
            for s in candidates
            if s.stage1_tile_shape == s1_shape
            and s.stage2_tile_shape == s2_shape
            and (policy1 is None or s.stage1_weight_load_policy == policy1)
            and (policy2 is None or s.stage2_weight_load_policy == policy2)
        )
        if not matches:
            raise ValueError("no implementation for requested tile/weight_load_policy combination")
        selected = min(
            matches,
            key=lambda s: (
                s.stage1_weight_load_policy != (policy1 if policy1 is not None else preferred1),
                s.stage2_weight_load_policy != (policy2 if policy2 is not None else preferred2),
            ),
        )
    return MoeConfig(selected, requested.experts, requested.topk)


def get_2stage_cfgs(
    token: int,
    model_dim: int,
    inter_dim: int,
    expert: int,
    topk: int,
    *,
    activation: ActivationFunction | str,
    bias_dtype: DataType | str,
    dtype: DataType | str = DataType.BF16,
    q_dtype_a: DataType | str = DataType.MXFP4,
    q_dtype_w: DataType | str = DataType.MXFP4,
    group_size: int = 32,
    arch: str = "gfx950",
    weight_load_policy: WeightLoadPolicy | str | None = None,
    stage1_weight_load_policy: WeightLoadPolicy | str | None = None,
    stage2_weight_load_policy: WeightLoadPolicy | str | None = None,
    solution_id: MoeSolutionId | int | None = None,
) -> MoeConfig:
    """Select one supported two-stage configuration for a problem.

    M participates in selection/cache lookup but not in the resulting kernel
    configuration. An explicit ID overrides automatic selection; conflicting
    shape, dtype, activation, bias or explicit policy requests are rejected.
    This entry returns two-stage configurations and has no one-stage fallback.
    """
    if type(token) is not int or token < 0:
        raise ValueError("token must be a nonnegative integer")
    requested, arch = _normalize_request(
        model_dim, inter_dim, activation, bias_dtype, dtype, q_dtype_a, q_dtype_w, group_size, arch
    )
    common_policy = _normalize_weight_load_policy(weight_load_policy)
    policy1 = _normalize_weight_load_policy(stage1_weight_load_policy)
    policy2 = _normalize_weight_load_policy(stage2_weight_load_policy)
    if common_policy is not None:
        if any(p is not None and p != common_policy for p in (policy1, policy2)):
            raise ValueError("common and stage-specific weight load policies conflict")
        policy1 = policy2 = common_policy
    if solution_id is not None and (not isinstance(solution_id, MoeSolutionId)):
        solution_id = MoeSolutionId.from_int(solution_id)
    config = MoeConfig(requested, expert, topk)
    return _get_2stage_cfgs_cached(token, config, arch, policy1, policy2, solution_id)


get_2stage_cfgs.cache_clear = _get_2stage_cfgs_cached.cache_clear
get_2stage_cfgs.cache_info = _get_2stage_cfgs_cached.cache_info
