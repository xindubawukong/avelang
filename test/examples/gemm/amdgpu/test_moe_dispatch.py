"""Two-stage selection preserves problem semantics and complete cache keys."""

from dataclasses import replace
from enum import IntEnum

import pytest
import torch
from avelang_kernels.amdgpu.local_moe import (
    MoeConfig,
    available_2stage_solutions,
    dispatch,
    dynamic_mxfp4_moe,
    get_2stage_cfgs,
)
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
from avelang_kernels.amdgpu.local_moe.stage1 import make_stage1
from avelang_kernels.amdgpu.local_moe.stage2 import make_stage2

PROBLEM = {"model_dim": 256, "inter_dim": 256, "expert": 16, "topk": 4, "activation": "swiglu", "bias_dtype": "bf16"}


def choose(token=8, **overrides):
    return get_2stage_cfgs(token=token, **PROBLEM | overrides)


@pytest.mark.parametrize("token,policy", [(0, 1), (128, 1), (255, 1), (256, 0), (257, 0), (1024, 0)])
def test_default_policy_threshold(token, policy):
    config = choose(token)
    assert (
        config.solution.stage1_weight_load_policy
        == config.solution.stage2_weight_load_policy
        == WeightLoadPolicy(policy)
    )
    assert config.solution.stages == Stages.TWO_STAGE
    assert config.stage2_workers == 256
    assert not hasattr(config, "token")


def test_available_candidates_and_explicit_policy():
    candidates = available_2stage_solutions(**PROBLEM)
    assert {s.stage1_weight_load_policy for s in candidates} == set(WeightLoadPolicy)
    assert {s.stage2_weight_load_policy for s in candidates} == set(WeightLoadPolicy)
    assert {s.stage1_tile_shape for s in candidates} == {Stage1TileShape.M32_N256, Stage1TileShape.M64_N512}
    for solution in candidates:
        assert choose(1024, solution_id=int(solution)).solution == solution
    for policy in WeightLoadPolicy:
        selected = choose(weight_load_policy=policy).solution
        assert selected.stage1_weight_load_policy == selected.stage2_weight_load_policy == policy


def test_selection_cache_normalizes_external_values():

    class ExternalActivation(IntEnum):
        SWIGLU = 19

    get_2stage_cfgs.cache_clear()
    first = choose()
    assert (
        choose(
            activation=ExternalActivation.SWIGLU,
            bias_dtype=DataType.BF16,
            dtype=torch.bfloat16,
            q_dtype_a=torch.uint8,
            q_dtype_w="fp4x2",
            arch="gfx950:sramecc+:xnack-",
            weight_load_policy="AUTO",
        )
        is first
    )
    assert get_2stage_cfgs.cache_info().hits == 1
    get_2stage_cfgs.cache_clear()
    assert get_2stage_cfgs.cache_info().currsize == 0


def test_kernel_factory_cache_uses_experts_and_topk_but_not_token():
    base = choose(weight_load_policy="cached")
    other_m = choose(16, weight_load_policy="cached")
    other_e = choose(expert=32, weight_load_policy="cached")
    other_k = choose(topk=2, weight_load_policy="cached")
    assert base.solution == other_e.solution == other_k.solution
    assert base == other_m and base != other_e and (base != other_k)
    for factory in (make_stage1, make_stage2):
        assert factory(base) is factory(other_m)
        assert factory(base) is not factory(other_e)
        assert factory(base) is not factory(other_k)


@pytest.mark.parametrize("field", ["expert", "topk"])
@pytest.mark.parametrize("invalid", [True, 1.0])
def test_invalid_problem_types_cannot_hit_warm_cache(field, invalid):
    choose(expert=1, topk=1)
    with pytest.raises(TypeError, match="integers"):
        choose(**{"expert": 1, "topk": 1} | {field: invalid})


@pytest.mark.parametrize("token", [-1, True, 8.0])
def test_invalid_token(token):
    with pytest.raises(ValueError, match="nonnegative integer"):
        choose(token)


@pytest.mark.parametrize(
    "overrides",
    [
        {"arch": "gfx942"},
        {"dtype": "mxfp4"},
        {"q_dtype_a": "bf16"},
        {"q_dtype_w": "nvfp4"},
        {"group_size": 64},
        {"activation": "relu"},
        {"bias_dtype": "mxfp4"},
        {"model_dim": 320},
        {"inter_dim": 16384},
        {"expert": 0},
        {"topk": 17},
    ],
)
def test_unsupported_requests(overrides):
    with pytest.raises(ValueError):
        choose(**overrides)
    with pytest.raises(ValueError):
        available_2stage_solutions(**PROBLEM | overrides)


@pytest.mark.parametrize(
    "field,value",
    [
        ("hidden", 512),
        ("intermediate", 512),
        ("activation", ActivationFunction.SILU_DOT),
        ("bias_dtype", DataType.NONE),
    ],
)
def test_explicit_id_must_match_request(field, value):
    solution = replace(choose().solution, **{field: value})
    with pytest.raises(ValueError, match=f"does not match requested {field}"):
        choose(solution_id=int(solution))


def test_explicit_id_policy_conflicts_and_unsupported_stages():
    solution = choose().solution
    with pytest.raises(ValueError, match="weight_load_policy"):
        choose(solution_id=solution, weight_load_policy="cached")
    for changed in (
        replace(solution, stages=Stages.ONE_STAGE),
        replace(solution, stage1_tile_shape=Stage1TileShape.M32_N128_K2),
    ):
        with pytest.raises(ValueError, match="unsupported Ave local MoE solution"):
            choose(solution_id=changed)


def test_config_stores_solution_and_has_derived_readonly_fields():
    solution = choose().solution
    config = MoeConfig.from_solution(solution, experts=16, topk=4)
    assert config.solution is solution
    assert config.activation is solution.activation
    with pytest.raises(TypeError):
        MoeConfig(solution, 16, 4, stage2_workers=128)
    with pytest.raises((AttributeError, TypeError)):
        config.hidden = 512


@pytest.mark.parametrize(
    "changes",
    [
        {"act_dtype": DataType.BF16},
        {"weight_dtype": DataType.NVFP4},
        {"bias_dtype": DataType.MXFP4},
        {"weight_ordering": WeightOrdering.PETIT_MXFP4},
        {"mfma": MfmaShape.BF16_MXFP4},
        {"stages": Stages.ONE_STAGE},
        {"stage1_buffering": Stage1Buffering.SINGLE_BUFFER},
        {"stage1_tile_shape": Stage1TileShape.M32_N128_K2},
        {"hidden": 320},
    ],
)
def test_direct_construction_cannot_bypass_implementation_resolution(changes):
    base = choose()
    config = replace(base, solution=replace(base.solution, **changes))
    for factory in (make_stage1, make_stage2):
        with pytest.raises(ValueError, match="unsupported Ave local MoE solution"):
            factory(config)
    with pytest.raises(ValueError, match="unsupported Ave local MoE solution"):
        dynamic_mxfp4_moe(None, None, None, config)


def test_dispatch_routes_only_registered_complete_combinations(monkeypatch):
    base = choose(weight_load_policy="cached")
    solution = replace(base.solution, stage1_tile_shape=Stage1TileShape.M64_N512)
    config = replace(base, solution=solution)
    stage1, stage2 = (object(), object())
    registry = dispatch._registered_2stage_implementations(config.hidden, config.intermediate)
    with monkeypatch.context() as patch:
        patch.setitem(registry, solution, (lambda cfg: stage1, lambda cfg: stage2))
        unsupported = replace(solution, bias_dtype=DataType.NONE)
        patch.delitem(registry, unsupported)
        get_2stage_cfgs.cache_clear()
        make_stage1.cache_clear()
        make_stage2.cache_clear()
        try:
            assert choose(solution_id=int(solution)) == config
            assert make_stage1(config) is stage1
            assert make_stage2(config) is stage2
            assert solution in available_2stage_solutions(**PROBLEM)
            with pytest.raises(ValueError, match="unsupported Ave local MoE solution"):
                make_stage1(replace(config, solution=unsupported))
        finally:
            get_2stage_cfgs.cache_clear()
            make_stage1.cache_clear()
            make_stage2.cache_clear()


@pytest.mark.parametrize(
    "field,name",
    [
        ("activation", "swiglu"),
        ("dtype", "bfloat16"),
        ("q_dtype_a", "uint8"),
        ("q_dtype_w", "fp4x2"),
        ("bias_dtype", "bf16"),
        ("weight_load_policy", "cached"),
    ],
)
def test_normalization_rejects_objects_that_only_look_like_names(field, name):

    class NamedObject:
        def __init__(self):
            self.name = name

        def __str__(self):
            return name

    with pytest.raises(ValueError, match="unsupported"):
        choose(**{field: NamedObject()})


def test_explicit_names_and_integer_codes_keep_the_same_solution():
    expected = choose(weight_load_policy="non_temporal")
    assert choose(activation="OPENAI_SWIGLU", dtype="torch.bfloat16", bias_dtype="BF16") == expected
    assert choose(activation=1, bias_dtype=5, weight_load_policy=1) == expected
    assert choose(bias_dtype=None) == choose(bias_dtype=DataType.NONE)


@pytest.mark.parametrize("tokens", [0, 8, 12, 16, 17, 128, 1024, 1025, 2047, 2048, 2049, 4096, 8191, 8192, 16384])
def test_kimi_selection_boundaries(tokens):
    config = get_2stage_cfgs(tokens, 3584, 384, 896, 16, activation="situ", bias_dtype="none")
    solution = config.solution
    assert solution.stage1_weight_load_policy == (tokens <= 2048)
    assert solution.stage2_weight_load_policy == (16 < tokens <= 1024)
    assert solution.stage1_tile_shape == (Stage1TileShape.M64_N256 if tokens >= 2048 else Stage1TileShape.M32_N128_K2)
    assert solution.stage2_tile_shape == (
        Stage2TileShape.M64_N256_K128 if tokens >= 2048 else Stage2TileShape.M32_N256_K128
    )
    assert config.use_route_reduce == (tokens >= 8192)
    ep = get_2stage_cfgs(tokens, 3584, 384, 896, 16, activation="situ", bias_dtype="none", is_ep=True)
    assert ep.solution == solution and (not ep.use_route_reduce)


def test_independent_policies_and_reduction_require_complete_routes():
    config = choose(stage1_weight_load_policy="cached", stage2_weight_load_policy="non_temporal")
    assert config.stage1_weight_load_aux == 0 and config.stage2_weight_load_aux == 2
    with pytest.raises(ValueError, match="conflict"):
        choose(weight_load_policy="cached", stage2_weight_load_policy="non_temporal")
    with pytest.raises(ValueError, match="complete local"):
        get_2stage_cfgs(
            8192, 3584, 384, 896, 16, activation="situ", bias_dtype="none", is_ep=True, use_route_reduce=True
        )


def mapping_kernel(n_tiles, m_group, groups):
    import avelang
    import avelang.language as al
    from avelang_kernels.amdgpu.local_moe.workgroup import make_grouped_workgroup_mapping

    map_workgroup = make_grouped_workgroup_mapping(n_tiles, m_group, groups)
    NT = n_tiles

    @avelang.jit
    def kernel(out: al.Pointer(al.u32), grid_m: al.u32):
        bid = al.convert(al.block_id(0), al.u32)
        result = al.make_tensor(out, al.u32, al.make_layout((grid_m * NT, 2), (2, 1)))
        n, m = map_workgroup(bid, grid_m)
        result[bid, 0] = n
        result[bid, 1] = m

    return kernel


@pytest.mark.skipif(not torch.version.hip or not torch.cuda.is_available(), reason="Requires AMD GPU")
@pytest.mark.parametrize("n_tiles,m_group,groups", [(3, 4, 8), (14, 2, 4)])
def test_grouped_mapping_is_a_permutation_including_partial_groups(n_tiles, m_group, groups):
    kernel = mapping_kernel(n_tiles, m_group, groups)
    for grid_m in (1, 2, 3, 5, 7, 16, 17, 65):
        output = torch.empty((grid_m * n_tiles, 2), dtype=torch.int32, device="cuda")
        kernel[lambda grid_m=grid_m: ((grid_m * n_tiles, 1, 1), (1, 1, 1))](output, grid_m)
        expected = []
        blocks = grid_m * n_tiles
        for block in range(blocks):
            group = block % groups
            remapped = group * (blocks // groups) + min(group, blocks % groups) + block // groups
            first_m = remapped // (m_group * n_tiles) * m_group
            height = min(grid_m - first_m, m_group)
            within = remapped % (m_group * n_tiles)
            expected.append((within // height, first_m + within % height))
        assert len(set(expected)) == blocks
        assert output.cpu().tolist() == [list(pair) for pair in expected]
