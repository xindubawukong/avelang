"""Layout and numerical contracts for the dynamic MXFP4 local MoE port."""

import pytest
import torch
from avelang_kernels.amdgpu.local_moe import get_2stage_cfgs
from avelang_kernels.amdgpu.local_moe.api import _pack_bias, _pack_weights
from avelang_kernels.amdgpu.local_moe.intermediate_mxfp4 import IntermediateLayout
from avelang_kernels.amdgpu.local_moe.scale_layout import scale_byte_shape, unsort_scales
from avelang_kernels.amdgpu.local_moe.solutionid import ActivationFunction


def quantize_reference(x, petit_intermediate=False):
    x = x.cpu().float().reshape(x.shape[0], -1, 32)
    maximum = x.abs().amax(-1)
    if petit_intermediate:
        bits = maximum.contiguous().view(torch.int32) + 4194304 & 4286578688
        exponent = (bits >> 23).clamp_min(2) - 2
    else:
        bits = (maximum * (1.0 / 6.0)).contiguous().view(torch.int32)
        exponent = (bits >> 23 & 255) + (bits >> 23 & 255 < 255).int() * (bits & 8388607 != 0).int()
    divisor = (exponent << 23).contiguous().view(torch.float32)
    scaled = x.abs() / torch.where(divisor == 0, 1.0, divisor)[..., None]
    levels = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    distances = (scaled[..., None] - levels).abs()
    codes = distances.argmin(-1)
    upper = (codes + 1).clamp_max(7)
    tie = distances.gather(-1, codes[..., None]) == distances.gather(-1, upper[..., None])
    codes += (codes & 1 != 0).int() * tie.squeeze(-1).int() * (codes != 7).int()
    codes |= torch.signbit(x).int() << 3
    codes = codes.reshape(x.shape[0], -1)
    act = (codes[:, ::2] | codes[:, 1::2] << 4).to(torch.uint8)
    return (act, exponent.to(torch.uint8))


def test_workspace_layout():
    layout = IntermediateLayout(96, 3072)
    workspace = torch.full((layout.nbytes,), 173, dtype=torch.uint8)
    act, scales = layout.views(workspace, tokens=8, topk=4)
    assert act.shape == (8, 4, 1536)
    assert scales.shape == scale_byte_shape(256, 3072)
    assert scales.data_ptr() - workspace.data_ptr() == 96 * 1536
    act.fill_(1)
    scales.fill_(2)
    assert torch.all(workspace[act.numel() : layout.scale_offset] == 173)


@pytest.mark.parametrize("k", [128, 256, 384, 512])
def test_native_weight_layout(k):
    e, n = (2, 256)
    data = torch.arange(e * n * k // 2).to(torch.uint8).reshape(e, n, k // 2)
    scales = torch.arange(e * n * k // 32).to(torch.uint8).reshape(e, n, k // 32)
    packed, ps = _pack_weights(data, scales)
    unpacked = packed.view(torch.int32).reshape(e * n // 256, 4, 4, k // 128, 4, 16, 4)
    unpacked = unpacked.permute(0, 1, 2, 5, 3, 4, 6).contiguous().view(torch.uint8).reshape_as(data)
    assert torch.equal(unpacked, data)
    padded_columns = (k + 255) // 256 * 8
    flat = ps.reshape(e * n // 32, (k + 255) // 256, 4, 16, 2, 2)
    unpacked_scales = flat.permute(0, 5, 3, 1, 4, 2).contiguous().reshape(e, n, padded_columns)
    assert torch.equal(unpacked_scales[..., : k // 32], scales)
    assert torch.all(unpacked_scales[..., k // 32 :] == 127)


def test_bias_layout():
    bias = torch.arange(2 * 2 * 512, dtype=torch.float32).to(torch.bfloat16).reshape(2, 2, 512)
    assert torch.equal(_pack_bias(_pack_bias(bias)), bias)


gfx950 = (
    bool(torch.version.hip and torch.cuda.is_available())
    and torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName.split(":")[0] == "gfx950"
)


def make_problem(config, tokens):
    from avelang_kernels.amdgpu.local_moe import ExpertWeights, Routing

    torch.manual_seed(71)
    e, d, i, topk = (config.experts, config.hidden, config.intermediate, config.topk)
    block_m = config.stage1_tile_m
    x = torch.randn((tokens, d), dtype=torch.bfloat16, device="cuda")
    w13 = torch.randint(0, 256, (e, 2 * i, d // 2), dtype=torch.uint8, device="cuda")
    w2 = torch.randint(0, 256, (e, d, i // 2), dtype=torch.uint8, device="cuda")
    s13 = torch.randint(117, 123, (e, 2 * i, d // 32), dtype=torch.uint8, device="cuda")
    s2 = torch.randint(117, 123, (e, d, i // 32), dtype=torch.uint8, device="cuda")
    b1 = torch.randn((e, 2, i), dtype=torch.bfloat16, device="cuda") if config.bias else None
    b2 = torch.randn((e, d), dtype=torch.bfloat16, device="cuda") if config.bias else None
    raw = (w13, w2, s13, s2, b1, b2)
    packed = ExpertWeights.pack(*raw)
    ids, route_weights, experts, routes = ([], [], [], [])
    for expert in range(e):
        begin = len(ids)
        for token in range(tokens):
            for slot in range(topk):
                if (token + slot) % e == expert:
                    ids.append(slot << 24 | token)
                    weight = (slot + 1) / (topk * (topk + 1) / 2)
                    route_weights.append(weight)
                    routes.append((token, slot, expert, weight))
        count = len(ids) - begin
        padding = -count % block_m
        ids += [topk << 24 | tokens] * padding
        route_weights += [0.0] * padding
        experts += [expert] * ((count + block_m - 1) // block_m)
    routing = Routing(
        torch.tensor(ids, dtype=torch.int32, device="cuda"),
        torch.tensor(route_weights, dtype=torch.float32, device="cuda"),
        torch.tensor(experts, dtype=torch.int32, device="cuda"),
        torch.tensor([len(ids), tokens], dtype=torch.int32, device="cuda"),
    )
    return (x, packed, routing, raw, routes)


def dequantize_reference(act, scales):
    act, scales = (act.cpu(), scales.cpu())
    codes = torch.stack((act & 15, act >> 4), -1).flatten(-2).long()
    levels = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])
    return levels[codes] * (2.0 ** (scales.float() - 127)).repeat_interleave(32, -1)


def moe_reference(x, config, raw, routes):
    w13, w2, s13, s2, b1, b2 = raw
    w13, w2 = (dequantize_reference(w13, s13), dequantize_reference(w2, s2))
    x = dequantize_reference(*quantize_reference(x))
    out = torch.zeros_like(x, dtype=torch.bfloat16)
    for token, slot, expert, weight in routes:
        gate, up = (w13[expert] @ x[token]).reshape(2, config.intermediate)
        if b1 is not None:
            gate = gate + b1[expert, 0].float().cpu()
            up = up + b1[expert, 1].float().cpu()
        if config.activation == ActivationFunction.OPENAI_SWIGLU:
            gate = gate.clamp_max(7)
            up = up.clamp(-7, 7) + 1
            hidden = gate * torch.sigmoid(1.702 * gate) * up
        elif config.activation == ActivationFunction.SITU_V2:
            hidden = 100 * torch.tanh(gate * 0.25) * torch.sigmoid(gate) * torch.tanh(up * 0.04)
        else:
            hidden = torch.nn.functional.silu(gate) * up
        hidden = dequantize_reference(*quantize_reference(hidden[None], petit_intermediate=True))[0]
        result = w2[expert] @ hidden
        if b2 is not None:
            result = result + b2[expert].float().cpu()
        out[token] += (result * weight).bfloat16()
    return out


@pytest.mark.skipif(not gfx950, reason="Native MXFP4 requires gfx950")
@pytest.mark.parametrize("tokens", [1, 7, 33])
@pytest.mark.parametrize("activation,bias", [("silu", False), ("swiglu", True)])
def test_dynamic_moe_reference(tokens, activation, bias):
    from avelang_kernels.amdgpu.local_moe import dynamic_mxfp4_moe

    config = get_2stage_cfgs(tokens, 256, 256, 3, 2, activation=activation, bias_dtype="bf16" if bias else "none")
    x, weights, routing, raw, routes = make_problem(config, tokens)
    actual = dynamic_mxfp4_moe(x, weights, routing, config).cpu()
    expected = moe_reference(x, config, raw, routes)
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.01)


@pytest.mark.skipif(not gfx950, reason="Native MXFP4 requires gfx950")
def test_moe_graph_replay_updates_inputs_and_clears_output():
    from avelang_kernels.amdgpu.local_moe import MoeWorkspace, dynamic_mxfp4_moe

    config = get_2stage_cfgs(7, 256, 256, 3, 2, activation="swiglu", bias_dtype="bf16")
    x, weights, routing, raw, routes = make_problem(config, 7)
    workspace = MoeWorkspace.allocate(x, routing, config)
    dynamic_mxfp4_moe(x, weights, routing, config, workspace=workspace)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        dynamic_mxfp4_moe(x, weights, routing, config, workspace=workspace)
    x.mul_(2)
    routing.weights.mul_(0.5)
    expected = moe_reference(x, config, raw, [(t, s, e, w * 0.5) for t, s, e, w in routes])
    for _ in range(3):
        graph.replay()
        torch.testing.assert_close(workspace.out.cpu(), expected, rtol=0.02, atol=0.01)


@pytest.mark.skipif(not gfx950, reason="Native MXFP4 requires gfx950")
@pytest.mark.parametrize("bias1,bias2", [(False, False), (True, False), (False, True)])
def test_independent_optional_biases(bias1, bias2):
    from dataclasses import replace

    from avelang_kernels.amdgpu.local_moe import dynamic_mxfp4_moe

    config = get_2stage_cfgs(3, 256, 256, 3, 2, activation="swiglu", bias_dtype="bf16")
    x, weights, routing, raw, routes = make_problem(config, 3)
    weights = replace(weights, bias1=weights.bias1 if bias1 else None, bias2=weights.bias2 if bias2 else None)
    raw = (*raw[:4], raw[4] if bias1 else None, raw[5] if bias2 else None)
    actual = dynamic_mxfp4_moe(x, weights, routing, config)
    expected = moe_reference(x, config, raw, routes)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0.02, atol=0.01)


@pytest.mark.skipif(not gfx950, reason="Native MXFP4 requires gfx950")
def test_graph_replay_reads_changed_routing_and_empty_extent():
    from avelang_kernels.amdgpu.local_moe import MoeWorkspace, dynamic_mxfp4_moe

    config = get_2stage_cfgs(7, 256, 256, 3, 2, activation="silu", bias_dtype="none")
    x, weights, routing, raw, routes = make_problem(config, 7)
    workspace = MoeWorkspace.allocate(x, routing, config)
    dynamic_mxfp4_moe(x, weights, routing, config, workspace=workspace)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        dynamic_mxfp4_moe(x, weights, routing, config, workspace=workspace)
    valid = routing.ids & 16777215 < len(x)
    routing.ids[valid] = routing.ids[valid] & 4278190080 | ((routing.ids[valid] & 16777215) + 1) % len(x)
    routing.experts.add_(1).remainder_(config.experts)
    graph.replay()
    expected = moe_reference(
        x, config, raw, [((t + 1) % len(x), s, (e + 1) % config.experts, w) for t, s, e, w in routes]
    )
    torch.testing.assert_close(workspace.out.cpu(), expected, rtol=0.02, atol=0.01)
    routing.counts[0] = 0
    graph.replay()
    assert torch.count_nonzero(workspace.out).item() == 0


@pytest.mark.skipif(not gfx950, reason="Native MXFP4 requires gfx950")
@pytest.mark.parametrize("case", ["empty", "zero_weights", "invalid_experts"])
def test_no_contributing_routes(case):
    from avelang_kernels.amdgpu.local_moe import dynamic_mxfp4_moe

    config = get_2stage_cfgs(0 if case == "empty" else 3, 256, 256, 3, 2, activation="silu", bias_dtype="none")
    x, weights, routing, _, _ = make_problem(config, 0 if case == "empty" else 3)
    if case == "zero_weights":
        routing.weights.zero_()
    if case == "invalid_experts":
        routing.experts.fill_(-1)
    output = dynamic_mxfp4_moe(x, weights, routing, config)
    assert output.shape == x.shape
    assert torch.count_nonzero(output).item() == 0


@pytest.mark.skipif(not gfx950, reason="Native MXFP4 requires gfx950")
@pytest.mark.parametrize("policy", ["cached", "non_temporal"])
def test_persistent_stage2_reuses_worker_after_invalid_group(policy):
    from avelang_kernels.amdgpu.local_moe.stage2 import make_stage2

    groups, d, i = (513, 256, 512)
    capacity, tokens = (groups * 32, groups * 32 - 7)
    config = get_2stage_cfgs(tokens, d, i, 1, 1, activation="silu", bias_dtype="none", weight_load_policy=policy)
    layout = IntermediateLayout(capacity, i)
    workspace = torch.empty(layout.nbytes, device="cuda", dtype=torch.uint8)
    act, scales = layout.views(workspace, tokens, 1)
    act.fill_(34)
    scales.fill_(127)
    weight, ws = _pack_weights(
        torch.full((1, d, i // 2), 34, device="cuda", dtype=torch.uint8),
        torch.full((1, d, i // 32), 127, device="cuda", dtype=torch.uint8),
    )
    ids = torch.full((capacity,), tokens, device="cuda", dtype=torch.int32)
    ids[:tokens] = torch.arange(tokens - 1, -1, -1, device="cuda", dtype=torch.int32)
    experts = torch.zeros(groups, device="cuda", dtype=torch.int32)
    experts[0] = -1
    route_weights = torch.full((capacity,), 0.5, device="cuda")
    counts = torch.tensor([capacity, tokens], device="cuda", dtype=torch.int32)
    output = torch.zeros(tokens * d + 16, device="cuda", dtype=torch.bfloat16)
    output[-16:] = 19
    make_stage2(config)[lambda: ((1, 256, 1), (256, 1, 1))](
        workspace, weight, ws, output, ids, experts, route_weights, counts, output, capacity, num_warps=4
    )
    expected = torch.full((tokens, d), i * 0.5, device="cuda", dtype=torch.bfloat16)
    expected[-32:] = 0
    torch.testing.assert_close(output[:-16].reshape(tokens, d), expected, rtol=0, atol=0)
    torch.testing.assert_close(output[-16:], torch.full_like(output[-16:], 19), rtol=0, atol=0)


@pytest.mark.skipif(not gfx950, reason="Native MXFP4 requires gfx950")
@pytest.mark.parametrize(
    "tokens,columns,capacity",
    [(7, 256, 64), (7, 3072, 64), (7, 7168, 2112), (1025, 256, 2112), (1025, 3072, 2112), (1025, 7168, 2112)],
)
def test_aiter_input_preparation(tokens, columns, capacity):
    from avelang_kernels.amdgpu.local_moe import MoeWorkspace, Routing
    from avelang_kernels.amdgpu.local_moe.api import prepare_input

    torch.manual_seed(67)
    x = torch.randn((tokens, columns), device="cuda", dtype=torch.bfloat16)
    x[0, :32] = 0
    topk, stride = (2, (tokens + 31) // 32 * 32)
    ids = torch.full((capacity,), tokens | topk << 24, device="cuda", dtype=torch.int32)
    for slot in range(topk):
        ids[slot * stride : slot * stride + tokens] = (
            torch.arange(tokens, device="cuda", dtype=torch.int32) | slot << 24
        )
    counts = torch.tensor([stride * topk, tokens], device="cuda", dtype=torch.int32)
    config = get_2stage_cfgs(tokens, columns, 256, 2, topk, activation="silu", bias_dtype="none")
    routing = Routing(
        ids, torch.ones(capacity, device="cuda"), torch.zeros(capacity // 32, device="cuda", dtype=torch.int32), counts
    )
    workspace = MoeWorkspace.allocate(x, routing, config)
    act, scales = prepare_input(x, routing, config, workspace)
    assert act.data_ptr() == workspace.input_act.data_ptr()
    assert scales.data_ptr() == workspace.sorted_scales.data_ptr()
    expected_act, expected_scales = quantize_reference(x)
    expected_scales[0, 0] = 92
    torch.testing.assert_close(act.cpu(), expected_act, rtol=0, atol=0)
    logical_scales = unsort_scales(scales, capacity, columns).cpu()
    valid = ids.cpu() & 16777215 < tokens
    token_ids = (ids.cpu()[valid] & 16777215).long()
    torch.testing.assert_close(logical_scales[valid], expected_scales[token_ids], rtol=0, atol=0)


@pytest.mark.skipif(not gfx950, reason="Native MXFP4 requires gfx950")
@pytest.mark.parametrize("s1,s2", [(0, 1), (4, 1), (2, 1), (2, 2)])
def test_k128_tiles_with_known_projection_and_padded_routes(s1, s2):
    from avelang_kernels.amdgpu.local_moe import ExpertWeights, MoeConfig, MoeSolutionId, dynamic_mxfp4_moe
    from avelang_kernels.amdgpu.local_moe.solutionid import DataType

    config = MoeConfig(
        MoeSolutionId(512, 384, ActivationFunction.SITU_V2, DataType.NONE, stage1_tile_shape=s1, stage2_tile_shape=s2),
        3,
        2,
    )
    x, weights, routing, _, routes = make_problem(config, 35)
    x.fill_(1)
    weights = ExpertWeights.pack(
        torch.full_like(weights.w13, 34),
        torch.full_like(weights.w2, 34),
        torch.full_like(weights.s13, 121),
        torch.full((3, 512, 12), 121, device="cuda", dtype=torch.uint8),
    )
    actual = dynamic_mxfp4_moe(x, weights, routing, config)
    gate = torch.tensor(8.0)
    activation = 100 * torch.tanh(gate / 4) * torch.sigmoid(gate) * torch.tanh(gate / 25)
    hidden = dequantize_reference(*quantize_reference(activation.expand(1, 384), petit_intermediate=True))[0]
    value = hidden.sum() / 64
    expected = torch.zeros((35, 512), dtype=torch.bfloat16)
    for token, slot, expert, weight in routes:
        expected[token] += (value * weight).bfloat16()
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.skipif(not gfx950, reason="Native MXFP4 requires gfx950")
@pytest.mark.parametrize("tile_m", [32, 64])
@pytest.mark.parametrize("route_output", [False, True])
def test_k128_stage2_nonuniform_rows_and_columns(tile_m, route_output):
    from avelang_kernels.amdgpu.local_moe import MoeConfig, MoeSolutionId
    from avelang_kernels.amdgpu.local_moe.solutionid import DataType, Stage1TileShape, Stage2TileShape
    from avelang_kernels.amdgpu.local_moe.stage2 import make_stage2

    tokens, hidden, intermediate, topk = (97, 3584, 384, 2)
    config = MoeConfig(
        MoeSolutionId(
            hidden,
            intermediate,
            ActivationFunction.SITU_V2,
            DataType.NONE,
            stage1_tile_shape=Stage1TileShape.M64_N256 if tile_m == 64 else Stage1TileShape.M32_N256,
            stage2_tile_shape=Stage2TileShape.M64_N256_K128 if tile_m == 64 else Stage2TileShape.M32_N256_K128,
        ),
        3,
        topk,
        route_output,
    )
    _, _, routing, raw, _ = make_problem(config, tokens)
    capacity = routing.capacity
    routing.weights.copy_((torch.arange(capacity, device="cuda") * 5 % 11 + 1).float() / 8)
    act = torch.randint(0, 256, (capacity, intermediate // 2), dtype=torch.uint8, device="cuda")
    rows = torch.arange(capacity, device="cuda")[:, None]
    columns = torch.arange(intermediate // 32, device="cuda")[None, :]
    scales = (125 + (rows + columns) % 3).to(torch.uint8)
    weight_scales = torch.full_like(raw[3], 125)
    w2, s2 = _pack_weights(raw[1], weight_scales)
    layout = IntermediateLayout(capacity, intermediate, sorted_act=True)
    workspace = torch.empty(layout.nbytes, dtype=torch.uint8, device="cuda")
    stored_act, stored_scales = layout.views(workspace, tokens, topk)
    stored_act.copy_(act)
    padded_scales = torch.full(layout.scale_shape, 127, dtype=torch.uint8, device="cuda")
    padded_scales[:capacity, : intermediate // 32] = scales
    stored_scales.copy_(padded_scales.reshape(-1, 2, 16, 2, 2, 4).permute(0, 3, 5, 2, 4, 1))
    out = torch.zeros((tokens, topk, hidden) if route_output else (tokens, hidden), dtype=torch.bfloat16, device="cuda")
    make_stage2(config)[lambda: config.stage2_grid(tokens, capacity)](
        workspace,
        w2,
        s2,
        None,
        routing.ids,
        routing.experts,
        routing.weights,
        routing.counts,
        out,
        capacity,
        num_warps=config.stage2_num_warps,
    )
    logical_act = dequantize_reference(act, scales)
    logical_weights = dequantize_reference(raw[1], weight_scales)
    ids = routing.ids.cpu()
    token, slot = (ids & 16777215, ids >> 24)
    expert_ids = routing.experts.cpu().repeat_interleave(tile_m)
    route_weights = routing.weights.cpu()
    expected = torch.zeros((tokens, topk, hidden), dtype=torch.bfloat16)
    for expert in range(config.experts):
        valid = (expert_ids == expert) & (token < tokens) & (slot < topk)
        values = logical_act[valid] @ logical_weights[expert].T
        expected[token[valid], slot[valid]] = (values * route_weights[valid, None]).bfloat16()
    if not route_output:
        expected = expected.float().sum(dim=1).bfloat16()
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


@pytest.mark.skipif(not gfx950, reason="Native MXFP4 requires gfx950")
def test_route_reduction_uses_fp32_and_large_buffer_addresses():
    from avelang_kernels.amdgpu.local_moe.route_reduce import make_route_reduce

    tokens, topk, hidden = (38000, 16, 3584)
    required = tokens * topk * hidden * 2
    if torch.cuda.mem_get_info()[0] < required * 2:
        pytest.skip("large-address check requires 9 GiB free GPU memory")
    src = torch.empty((tokens, topk, hidden), dtype=torch.bfloat16, device="cuda")
    src.fill_(1)
    src[:, 0].fill_(256)
    src[:, -1].fill_(-256)
    src[-1, 1].fill_(3)
    dst = torch.empty((tokens, hidden), dtype=torch.bfloat16, device="cuda")
    make_route_reduce(hidden, topk)[lambda: ((tokens, 1, 1), (512, 1, 1))](src, dst, tokens, num_warps=8)
    expected = torch.full_like(dst, 14)
    expected[-1].fill_(16)
    torch.testing.assert_close(dst, expected, rtol=0, atol=0)


@pytest.mark.skipif(not gfx950, reason="Native MXFP4 requires gfx950")
@pytest.mark.parametrize("activation", ["swiglu", "situ"])
def test_m64_n512_stage1_bias_and_m32_stage2(activation):
    from dataclasses import replace

    from avelang_kernels.amdgpu.local_moe import dynamic_mxfp4_moe
    from avelang_kernels.amdgpu.local_moe.solutionid import Stage1TileShape

    config = get_2stage_cfgs(35, 512, 256, 3, 2, activation=activation, bias_dtype="bf16")
    config = replace(config, solution=replace(config.solution, stage1_tile_shape=Stage1TileShape.M64_N512))
    x, weights, routing, raw, routes = make_problem(config, 35)
    actual = dynamic_mxfp4_moe(x, weights, routing, config)
    expected = moe_reference(x, config, raw, routes)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0.02, atol=0.01)


@pytest.mark.skipif(not gfx950, reason="Native MXFP4 requires gfx950")
def test_kimi_route_output_graph_replay():
    from avelang_kernels.amdgpu.local_moe import MoeWorkspace, dynamic_mxfp4_moe

    config = get_2stage_cfgs(8192, 3584, 384, 16, 16, activation="situ", bias_dtype="none")
    x, weights, routing, _, _ = make_problem(config, 3)
    workspace = MoeWorkspace.allocate(x, routing, config)
    dynamic_mxfp4_moe(x, weights, routing, config, workspace=workspace)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        dynamic_mxfp4_moe(x, weights, routing, config, workspace=workspace)
    x.mul_(2)
    graph.replay()
    expected = dynamic_mxfp4_moe(x, weights, routing, config)
    torch.testing.assert_close(workspace.out, expected, rtol=0, atol=0)
    routing.weights.zero_()
    graph.replay()
    assert torch.count_nonzero(workspace.out) == 0
