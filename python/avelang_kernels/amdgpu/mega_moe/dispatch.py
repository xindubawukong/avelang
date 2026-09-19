"""Registered complete solutions and Petit's token-dependent producer selection."""

from dataclasses import replace
from functools import cache

from .config import MegaMoeConfig
from .solutionid import ActivationFunction, DataType, MegaMoeSolutionId, ProducerGeometry

_PROFILES = (
    *((r, 32, 4, 2880, 3072, ActivationFunction.OPENAI_SWIGLU, DataType.BF16, (0,)) for r in (2, 4, 8)),
    (8, 128, 4, 2880, 3072, ActivationFunction.OPENAI_SWIGLU, DataType.BF16, (0, 1, 2)),
    (8, 256, 8, 7168, 2048, ActivationFunction.SILU_DOT, DataType.NONE, (0, 2, 3)),
    (8, 384, 6, 7168, 3072, ActivationFunction.SILU_DOT, DataType.NONE, (0, 3)),
    (8, 896, 16, 3584, 3072, ActivationFunction.SITU_V2, DataType.NONE, (0,)),
)


def get_2stage_cfgs(tokens, world_size, experts, topk, hidden, intermediate, *, activation, bias_dtype):
    """Select producer and Stage1 geometry for this invocation's token count."""
    if isinstance(activation, str):
        try:
            activation = {
                "silu": ActivationFunction.SILU_DOT,
                "swiglu": ActivationFunction.OPENAI_SWIGLU,
                "kimi_situ": ActivationFunction.SITU_V2,
            }[activation]
        except KeyError:
            raise ValueError("activation must be 'silu', 'swiglu' or 'kimi_situ'") from None
    if isinstance(bias_dtype, str):
        try:
            bias_dtype = {"none": DataType.NONE, "bf16": DataType.BF16}[bias_dtype]
        except KeyError:
            raise ValueError("bias_dtype must be 'none' or 'bf16'") from None
    solution = MegaMoeSolutionId(world_size, experts, topk, hidden, intermediate, activation, bias_dtype)
    MegaMoeConfig.validate_tokens(tokens)
    e, activation = solution.experts, solution.activation
    geometry = ProducerGeometry.CTA56
    if activation == ActivationFunction.SITU_V2:
        geometry = ProducerGeometry.CTA56
    elif e > 56:
        if activation == ActivationFunction.SILU_DOT and (
            (e == 256 and 12 <= tokens < 1024) or (e == 384 and tokens < 512)
        ):
            geometry = ProducerGeometry.CTA192
        elif tokens < 12:
            geometry = ProducerGeometry.CTA128
        elif tokens < 24:
            geometry = ProducerGeometry.CTA64
    threshold = 128 if e == 256 else 256
    tile_m, num_warps = 32, 4
    if solution.world_size != 1 and tokens >= threshold:
        tile_m = 64
        num_warps = 4 if e == 128 and tokens < 1024 else 8
    config = MegaMoeConfig(
        replace(solution, producer_geometry=geometry),
        stage1_tile_m=tile_m,
        stage1_num_warps=num_warps,
    )
    resolve_implementation(config)
    return config


@cache
def registered_solutions():
    return tuple(
        MegaMoeSolutionId(r, e, k, d, i, act, bias, producer_geometry=ProducerGeometry(p))
        for r, e, k, d, i, act, bias, geometries in _PROFILES
        for p in geometries
    )


@cache
def _registered_implementations():
    return {solution: make_kernels for solution in registered_solutions()}


def resolve_implementation(config):
    implementations = _registered_implementations()
    try:
        return implementations[config.solution]
    except KeyError:
        raise ValueError(f"no MegaMoE implementation registered for solution {int(config.solution):#x}") from None


def make_kernels(config):
    from .route_output import make_combine_kernel
    from .stage1 import make_stage1
    from .stage2 import make_stage2
    from .synchronization import make_global_barrier
    from .token_shuffle_direct_push import make_count, make_plan, make_push

    return (
        make_count(config),
        make_plan(config),
        make_push(config),
        make_stage1(config),
        make_stage2(config),
        make_combine_kernel(config),
        make_global_barrier(config),
    )
