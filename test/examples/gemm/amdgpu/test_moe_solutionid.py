"""Local solution IDs match Petit's published profile encodings."""

from dataclasses import replace

import pytest
from avelang_kernels.amdgpu.local_moe import MoeConfig, MoeSolutionId
from avelang_kernels.amdgpu.local_moe.solutionid import (
    ActivationFunction,
    DataType,
    MfmaShape,
    Stage1Buffering,
    Stage1TileShape,
    Stage2TileShape,
    Stages,
    WeightLoadPolicy,
    WeightOrdering,
)

GPTOSS = MoeSolutionId(3072, 3072, ActivationFunction.OPENAI_SWIGLU, DataType.BF16)


@pytest.mark.parametrize(
    "hidden,intermediate,activation,bias,experts,topk,encoded",
    [
        (3072, 3072, ActivationFunction.OPENAI_SWIGLU, DataType.BF16, 16, 4, 0x3030918511),
        (7168, 2048, ActivationFunction.SILU_DOT, DataType.NONE, 33, 9, 0x2070818011),
        (7168, 3072, ActivationFunction.SILU_DOT, DataType.NONE, 49, 7, 0x3070818011),
    ],
)
def test_petit_profile_ids(hidden, intermediate, activation, bias, experts, topk, encoded):
    solution = MoeSolutionId(hidden, intermediate, activation, bias)
    config = MoeConfig(solution, experts, topk)
    assert config.solution is solution
    assert MoeConfig.from_solution(solution, experts=experts, topk=topk).solution is solution
    assert int(config.solution) == encoded
    assert MoeSolutionId.from_int(encoded) == config.solution
    assert MoeConfig.from_solution(encoded, experts=config.experts, topk=config.topk) == config
    assert (config.stage1_tile_m, config.stage1_projection_n, config.stage2_tile_n, config.stage1_num_warps) == (
        32,
        128,
        256,
        4,
    )


def test_policy_bias_and_activation_fields():
    config = MoeConfig(GPTOSS, 16, 4)
    changed = replace(
        config,
        solution=replace(
            config.solution,
            activation=ActivationFunction.SILU_DOT,
            bias_dtype=DataType.NONE,
            stage1_weight_load_policy=WeightLoadPolicy.NON_TEMPORAL,
            stage2_weight_load_policy=WeightLoadPolicy.NON_TEMPORAL,
        ),
    )
    assert int(changed.solution) == 0x33030818011
    assert changed.stage1_weight_load_aux == changed.stage2_weight_load_aux == 2
    assert config.stage1_weight_load_aux == config.stage2_weight_load_aux == 0
    # Expert count and top-k are not encoded in the local solution ID.
    assert replace(config, experts=32, topk=8).solution == config.solution


@pytest.mark.parametrize("hidden,intermediate", [(64, 64), (16320, 16320)])
def test_shape_encoding_boundaries(hidden, intermediate):
    solution = MoeSolutionId(hidden, intermediate, ActivationFunction.SILU_DOT, DataType.NONE)
    assert MoeSolutionId.from_int(int(solution)) == solution


@pytest.mark.parametrize("shape", [0, 65, 16384])
def test_unencodable_dimensions(shape):
    with pytest.raises(ValueError, match="positive multiple of 64"):
        replace(GPTOSS, hidden=shape)
    with pytest.raises(ValueError, match="positive multiple of 64"):
        replace(GPTOSS, intermediate=shape)


@pytest.mark.parametrize("encoded", [-1, (1 << 47) | 0x3030918511, 1 << 63, 1 << 64, 0, 0x303091851F])
def test_invalid_ids(encoded):
    with pytest.raises(ValueError):
        MoeSolutionId.from_int(encoded)


@pytest.mark.parametrize("encoded", [True, 1.0, "0x3030918511"])
def test_id_requires_an_integer(encoded):
    with pytest.raises(TypeError):
        MoeSolutionId.from_int(encoded)


@pytest.mark.parametrize(
    "field,value",
    [
        ("act_dtype", DataType.BF16),
        ("weight_dtype", DataType.NVFP4),
        ("bias_dtype", DataType.MXFP4),
        ("weight_ordering", WeightOrdering.PETIT_MXFP4),
        ("mfma", MfmaShape.BF16_MXFP4),
        ("stages", Stages.ONE_STAGE),
        ("stage1_buffering", Stage1Buffering.SINGLE_BUFFER),
        ("stage1_tile_shape", Stage1TileShape.M64_N512),
    ],
)
def test_config_preserves_encodable_solutions(field, value):
    solution = replace(GPTOSS, **{field: value})
    assert MoeSolutionId.from_int(int(solution)) == solution
    config = MoeConfig.from_solution(solution, experts=16, topk=4)
    assert config.solution is solution
    assert config.bias == (solution.bias_dtype != DataType.NONE)


def test_stage1_m64_bit_and_shape():
    solution = MoeSolutionId.from_int(0x43030918511)
    assert solution.stage1_tile_shape == Stage1TileShape.M64_N512
    assert (solution.stage1_tile_m, solution.stage1_tile_n) == (64, 512)
    assert solution.stage1_weight_load_policy == solution.stage2_weight_load_policy == WeightLoadPolicy.CACHED


@pytest.mark.parametrize("s1,bits1", [(0, 0), (1, 1 << 42), (2, 1 << 44), (4, 1 << 46)])
@pytest.mark.parametrize("s2,bits2", [(0, 0), (1, 1 << 43), (2, 1 << 45)])
@pytest.mark.parametrize("policies", range(4))
def test_interleaved_tile_bits_and_independent_policies(s1, bits1, s2, bits2, policies):
    base = MoeSolutionId(3584, 384, ActivationFunction.SITU_V2, DataType.NONE)
    assert int(base) == 0x0638A18011
    encoded = int(base) | bits1 | bits2 | (policies << 40)
    solution = MoeSolutionId.from_int(encoded)
    assert solution.stage1_tile_shape == Stage1TileShape(s1)
    assert solution.stage2_tile_shape == Stage2TileShape(s2)
    assert solution.stage1_weight_load_policy == policies & 1
    assert solution.stage2_weight_load_policy == policies >> 1
    assert int(solution) == encoded


@pytest.mark.parametrize("bits", [(1 << 42) | (1 << 44), (1 << 43) | (1 << 45)])
def test_reserved_tile_values(bits):
    with pytest.raises(ValueError):
        MoeSolutionId.from_int(int(GPTOSS) | bits)


def test_config_keeps_shapes_independent_of_kernel_tile_constraints():
    solution = replace(GPTOSS, hidden=320)
    assert MoeConfig(solution, 16, 4).hidden == 320
