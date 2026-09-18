"""Layout and numerical contracts for the dynamic MXFP4 local MoE port."""

import pytest
import torch
from avelang_kernels.amdgpu.local_moe import get_2stage_cfgs
from avelang_kernels.amdgpu.local_moe.api import _pack_bias, _pack_weights
from avelang_kernels.amdgpu.local_moe.intermediate_mxfp4 import IntermediateLayout
from avelang_kernels.amdgpu.local_moe.scale_layout import unsort_scales
from avelang_kernels.amdgpu.local_moe.solutionid import ActivationFunction


def quantize_reference(x, petit_intermediate=False):
    x = x.cpu().float().reshape(x.shape[0], -1, 32)
    maximum = x.abs().amax(-1)
    if petit_intermediate:
        bits = (maximum.contiguous().view(torch.int32) + 0x00400000) & 0xFF800000
        exponent = (bits >> 23).clamp_min(2) - 2
    else:
        bits = (maximum * (1.0 / 6.0)).contiguous().view(torch.int32)
        exponent = ((bits >> 23) & 255) + (((bits >> 23) & 255) < 255).int() * ((bits & 0x7FFFFF) != 0).int()
    divisor = (exponent << 23).contiguous().view(torch.float32)
    scaled = x.abs() / torch.where(divisor == 0, 1.0, divisor)[..., None]
    levels = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    distances = (scaled[..., None] - levels).abs()
    codes = distances.argmin(-1)
    upper = (codes + 1).clamp_max(7)
    tie = distances.gather(-1, codes[..., None]) == distances.gather(-1, upper[..., None])
    codes += ((codes & 1) != 0).int() * tie.squeeze(-1).int() * (codes != 7).int()
    codes |= torch.signbit(x).int() << 3
    codes = codes.reshape(x.shape[0], -1)
    act = (codes[:, ::2] | (codes[:, 1::2] << 4)).to(torch.uint8)
    return act, exponent.to(torch.uint8)


def test_workspace_layout():
    layout = IntermediateLayout(96, 3072)
    workspace = torch.full((layout.nbytes,), 173, dtype=torch.uint8)
    act, scales = layout.views(workspace, tokens=8, topk=4)
    assert act.shape == (8, 4, 1536)
    assert scales.shape == (256, 96)
    assert scales.data_ptr() - workspace.data_ptr() == 96 * 1536
    act.fill_(1)
    scales.fill_(2)
    assert torch.all(workspace[act.numel() : layout.scale_offset] == 173)


@pytest.mark.parametrize("k", [256, 512])
def test_native_weight_layout(k):
    e, n = 2, 256
    data = torch.arange(e * n * k // 2).to(torch.uint8).reshape(e, n, k // 2)
    scales = torch.arange(e * n * k // 32).to(torch.uint8).reshape(e, n, k // 32)
    packed, ps = _pack_weights(data, scales)
    # Invert Petit's original seven-axis tile permutation independently.
    unpacked = packed.view(torch.int32).reshape(e * n // 256, 4, 4, k // 128, 4, 16, 4)
    unpacked = unpacked.permute(0, 1, 2, 5, 3, 4, 6).contiguous().view(torch.uint8).reshape_as(data)
    assert torch.equal(unpacked, data)
    flat = ps.reshape(e * n // 32, k // 256, 4, 16, 2, 2)
    unpacked_scales = flat.permute(0, 5, 3, 1, 4, 2).contiguous().reshape(e, n, k // 32)
    assert torch.equal(unpacked_scales, scales)


@pytest.mark.parametrize("k", [128, 384])
def test_native_weights_reject_partial_k256_tiles(k):
    data = torch.zeros((1, 256, k // 2), dtype=torch.uint8)
    scales = torch.full((1, 256, k // 32), 127, dtype=torch.uint8)
    with pytest.raises(ValueError, match="N and K divisible by 256"):
        _pack_weights(data, scales)


def test_bias_layout():
    bias = torch.arange(2 * 2 * 512, dtype=torch.float32).to(torch.bfloat16).reshape(2, 2, 512)
    # Bias packing is an involution: it swaps the middle two tile axes.
    assert torch.equal(_pack_bias(_pack_bias(bias)), bias)


gfx950 = bool(torch.version.hip and torch.cuda.is_available()) and (
    torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName.split(":")[0] == "gfx950"
)


def make_problem(config, tokens):
    from avelang_kernels.amdgpu.local_moe import ExpertWeights, Routing

    torch.manual_seed(71)
    e, d, i, topk = config.experts, config.hidden, config.intermediate, config.topk
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
    ids, route_weights, experts, routes = [], [], [], []
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
        padding = (-count) % block_m
        ids += [(topk << 24) | tokens] * padding
        route_weights += [0.0] * padding
        experts += [expert] * ((count + block_m - 1) // block_m)
    routing = Routing(
        torch.tensor(ids, dtype=torch.int32, device="cuda"),
        torch.tensor(route_weights, dtype=torch.float32, device="cuda"),
        torch.tensor(experts, dtype=torch.int32, device="cuda"),
        torch.tensor([len(ids), tokens], dtype=torch.int32, device="cuda"),
    )
    return x, packed, routing, raw, routes


def dequantize_reference(act, scales):
    act, scales = act.cpu(), scales.cpu()
    codes = torch.stack((act & 15, act >> 4), -1).flatten(-2).long()
    levels = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])
    return levels[codes] * (2.0 ** (scales.float() - 127)).repeat_interleave(32, -1)


def moe_reference(x, config, raw, routes):
    w13, w2, s13, s2, b1, b2 = raw
    w13, w2 = dequantize_reference(w13, s13), dequantize_reference(w2, s2)
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
    valid = (routing.ids & 0xFFFFFF) < len(x)
    routing.ids[valid] = (routing.ids[valid] & 0xFF000000) | (((routing.ids[valid] & 0xFFFFFF) + 1) % len(x))
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
    topk, stride = 2, ((tokens + 31) // 32) * 32
    ids = torch.full((capacity,), tokens | (topk << 24), device="cuda", dtype=torch.int32)
    for slot in range(topk):
        ids[slot * stride : slot * stride + tokens] = torch.arange(tokens, device="cuda", dtype=torch.int32) | (
            slot << 24
        )
    counts = torch.tensor([stride * topk, tokens], device="cuda", dtype=torch.int32)
    config = get_2stage_cfgs(tokens, columns, 256, 2, topk, activation="silu", bias_dtype="none")
    routing = Routing(
        ids,
        torch.ones(capacity, device="cuda"),
        torch.zeros(capacity // 32, device="cuda", dtype=torch.int32),
        counts,
    )
    workspace = MoeWorkspace.allocate(x, routing, config)
    act, scales = prepare_input(x, routing, config, workspace)
    assert act.data_ptr() == workspace.input_act.data_ptr()
    assert scales.data_ptr() == workspace.sorted_scales.data_ptr()
    expected_act, expected_scales = quantize_reference(x)
    # AITER input kernels seed amax with 1e-10, including all-zero blocks.
    expected_scales[0, 0] = 92
    torch.testing.assert_close(act.cpu(), expected_act, rtol=0, atol=0)
    logical_scales = unsort_scales(scales, capacity, columns).cpu()
    valid = (ids.cpu() & 0xFFFFFF) < tokens
    token_ids = (ids.cpu()[valid] & 0xFFFFFF).long()
    torch.testing.assert_close(logical_scales[valid], expected_scales[token_ids], rtol=0, atol=0)


@pytest.mark.skipif(not gfx950, reason="Native MXFP4 requires gfx950")
@pytest.mark.parametrize("policy", ["cached"])
def test_persistent_stage2_reuses_worker_after_invalid_group(policy):
    from avelang_kernels.amdgpu.local_moe.stage2 import make_stage2

    # Worker 0 owns groups 0, 1, 2: it must continue past invalid group 0.
    groups, d, i = 513, 256, 512
    capacity, tokens = groups * 32, groups * 32 - 7
    config = get_2stage_cfgs(tokens, d, i, 1, 1, activation="silu", bias_dtype="none", weight_load_policy=policy)
    layout = IntermediateLayout(capacity, i)
    workspace = torch.empty(layout.nbytes, device="cuda", dtype=torch.uint8)
    act, scales = layout.views(workspace, tokens, 1)
    act.fill_(0x22)  # Two FP4 ones per byte.
    scales.fill_(127)
    weight, ws = _pack_weights(
        torch.full((1, d, i // 2), 0x22, device="cuda", dtype=torch.uint8),
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
