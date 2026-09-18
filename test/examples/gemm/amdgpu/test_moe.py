"""End-to-end numerical correctness tests for dynamic MXFP4 LocalMoE."""

import pytest
import torch
from avelang_kernels.amdgpu.local_moe import (
    ExpertWeights,
    Routing,
    dynamic_mxfp4_moe,
    get_2stage_cfgs,
)

MXFP4 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def has_gfx950():
    return torch.cuda.is_available() and torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx950")


def quantize_reference(values, *, intermediate=False):
    rows = values.cpu().float().reshape(values.shape[0], -1, 32)
    maximum = rows.abs().amax(-1)
    if intermediate:
        bits = (maximum.contiguous().view(torch.int32) + 0x00400000) & 0xFF800000
        exponent = (bits >> 23).clamp_min(2) - 2
    else:
        bits = (maximum / 6.0).contiguous().view(torch.int32)
        exponent = ((bits >> 23) & 255) + (((bits >> 23) & 255) < 255).int() * ((bits & 0x7FFFFF) != 0).int()
    divisor = (exponent << 23).contiguous().view(torch.float32)
    scaled = rows.abs() / torch.where(divisor == 0, 1.0, divisor)[..., None]
    levels = MXFP4[:8]
    distances = (scaled[..., None] - levels).abs()
    codes = distances.argmin(-1)
    upper = (codes + 1).clamp_max(7)
    ties = distances.gather(-1, codes[..., None]) == distances.gather(-1, upper[..., None])
    codes += ((codes & 1) != 0).int() * ties.squeeze(-1).int() * (codes != 7).int()
    codes |= torch.signbit(rows).int() << 3
    codes = codes.reshape(values.shape[0], -1)
    packed = (codes[:, ::2] | (codes[:, 1::2] << 4)).to(torch.uint8)
    return packed, exponent.to(torch.uint8)


def dequantize(values, scales):
    values, scales = values.cpu(), scales.cpu()
    codes = torch.stack((values & 15, values >> 4), -1).flatten(-2).long()
    return MXFP4[codes] * torch.pow(2.0, scales.float() - 127.0).repeat_interleave(32, -1)


def make_problem(config, tokens, bias1, bias2):
    generator = torch.Generator(device="cuda").manual_seed(71 + tokens)
    experts, hidden, intermediate, topk = config.experts, config.hidden, config.intermediate, config.topk
    x = torch.randn((tokens, hidden), generator=generator, dtype=torch.bfloat16, device="cuda")
    w13 = torch.randint(
        0, 256, (experts, 2 * intermediate, hidden // 2), generator=generator, dtype=torch.uint8, device="cuda"
    )
    w2 = torch.randint(
        0, 256, (experts, hidden, intermediate // 2), generator=generator, dtype=torch.uint8, device="cuda"
    )
    s13 = torch.randint(
        117, 123, (experts, 2 * intermediate, hidden // 32), generator=generator, dtype=torch.uint8, device="cuda"
    )
    s2 = torch.randint(
        117, 123, (experts, hidden, intermediate // 32), generator=generator, dtype=torch.uint8, device="cuda"
    )
    b1 = (
        torch.randn((experts, 2, intermediate), generator=generator, dtype=torch.bfloat16, device="cuda")
        if bias1
        else None
    )
    b2 = torch.randn((experts, hidden), generator=generator, dtype=torch.bfloat16, device="cuda") if bias2 else None

    ids, weights, route_experts, routes = [], [], [], []
    for expert in range(experts):
        begin = len(ids)
        for token in range(tokens):
            for slot in range(topk):
                if (token + slot) % experts == expert:
                    weight = (slot + 1) / (topk * (topk + 1) / 2)
                    ids.append(slot << 24 | token)
                    weights.append(weight)
                    routes.append((token, expert, weight))
        padding = (begin - len(ids)) % config.stage1_tile_m
        ids.extend([(topk << 24) | tokens] * padding)
        weights.extend([0.0] * padding)
        route_experts.extend([expert] * ((len(ids) - begin) // config.stage1_tile_m))

    routing = Routing(
        torch.tensor(ids, dtype=torch.int32, device="cuda"),
        torch.tensor(weights, dtype=torch.float32, device="cuda"),
        torch.tensor(route_experts, dtype=torch.int32, device="cuda"),
        torch.tensor([len(ids), tokens], dtype=torch.int32, device="cuda"),
    )
    raw = (w13, w2, s13, s2, b1, b2)
    return x, ExpertWeights.pack(*raw), routing, raw, routes


def reference(x, config, raw, routes):
    w13, w2, s13, s2, bias1, bias2 = raw
    w13 = dequantize(w13, s13)
    w2 = dequantize(w2, s2)
    x = dequantize(*quantize_reference(x))
    output = torch.zeros_like(x, dtype=torch.bfloat16)
    for token, expert, weight in routes:
        gate, up = (w13[expert] @ x[token]).reshape(2, config.intermediate)
        if bias1 is not None:
            gate = gate + bias1[expert, 0].float().cpu()
            up = up + bias1[expert, 1].float().cpu()
        if config.activation.name == "OPENAI_SWIGLU":
            gate = gate.clamp_max(7.0)
            up = up.clamp(-7.0, 7.0) + 1.0
            activated = gate * torch.sigmoid(1.702 * gate) * up
        else:
            activated = torch.nn.functional.silu(gate) * up
        activated = dequantize(*quantize_reference(activated[None], intermediate=True))[0]
        result = w2[expert] @ activated
        if bias2 is not None:
            result = result + bias2[expert].float().cpu()
        output[token] += (result * weight).bfloat16()
    return output


def assert_matches_reference(actual, expected):
    actual, expected = actual.float(), expected.float()
    error = (actual - expected).abs()
    scale = expected.square().mean().sqrt().clamp_min(1.0e-12)
    normalized = error / scale
    assert torch.isfinite(actual).all()
    assert normalized.mean().item() < 0.005
    assert torch.quantile(normalized, 0.99).item() < 0.03
    assert normalized.max().item() < 0.20


@pytest.mark.gpu
@pytest.mark.skipif(not has_gfx950(), reason="native MXFP4 requires gfx950")
@pytest.mark.parametrize(
    "tokens,activation,bias1,bias2",
    [
        (1, "silu", False, False),
        (7, "silu", True, False),
        (7, "swiglu", False, True),
        (33, "swiglu", True, True),
    ],
)
def test_local_moe_matches_reference(tokens, activation, bias1, bias2):
    config = get_2stage_cfgs(
        tokens, 256, 256, 3, 2, activation=activation, bias_dtype="bf16" if bias1 or bias2 else "none"
    )
    x, weights, routing, raw, routes = make_problem(config, tokens, bias1, bias2)
    actual = dynamic_mxfp4_moe(x, weights, routing, config).cpu()
    expected = reference(x, config, raw, routes)
    assert_matches_reference(actual, expected)


@pytest.mark.gpu
@pytest.mark.skipif(not has_gfx950(), reason="native MXFP4 requires gfx950")
@pytest.mark.parametrize("empty,zero_weights", [(True, False), (False, True)])
def test_local_moe_no_contributing_routes_returns_zero(empty, zero_weights):
    tokens = 0 if empty else 7
    config = get_2stage_cfgs(tokens, 256, 256, 3, 2, activation="silu", bias_dtype="none")
    x, weights, routing, _, _ = make_problem(config, tokens, False, False)
    if zero_weights:
        routing.weights.zero_()
    actual = dynamic_mxfp4_moe(x, weights, routing, config)
    assert actual.shape == x.shape
    assert torch.count_nonzero(actual).item() == 0
