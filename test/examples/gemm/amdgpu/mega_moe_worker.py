"""Distributed numerical reference check for dynamic MXFP4 MegaMoE."""

import gc
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
from avelang_kernels.amdgpu.mega_moe import (
    MegaMoeWorkspace,
    dynamic_mxfp4_mega_moe,
    get_2stage_cfgs,
    pack_expert_weights,
    registered_solutions,
)

MXFP4 = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


@dataclass(frozen=True)
class Profile:
    name: str
    tokens: int
    experts: int
    topk: int
    hidden: int
    intermediate: int
    activation: str
    bias: bool


PROFILES = (
    Profile("gptoss", 7, 32, 4, 2880, 3072, "swiglu", True),
    Profile("kimi", 8, 896, 16, 3584, 3072, "kimi_situ", False),
)


def dequantize(values, scales):
    table = torch.tensor(MXFP4, dtype=torch.float32, device=values.device)
    unpacked = torch.empty((*values.shape[:-1], values.shape[-1] * 2), dtype=torch.float32, device=values.device)
    unpacked[..., 0::2] = table[(values & 0xF).long()]
    unpacked[..., 1::2] = table[(values >> 4).long()]
    scale = torch.pow(2.0, scales.float() - 127.0)
    return (unpacked.view(*unpacked.shape[:-1], scales.shape[-1], 32) * scale.unsqueeze(-1)).flatten(-2)


def quantize_dequantize(values, *, intermediate=False):
    rows = values.to(torch.bfloat16).float().view(values.shape[0], -1, 32)
    maximum = rows.abs().amax(dim=-1)
    if intermediate:
        bits = (maximum.contiguous().view(torch.int32) + 0x00400000) & 0xFF800000
        exponent = (bits >> 23).clamp_min(2) - 2
    else:
        bits = (maximum / 6.0).contiguous().view(torch.int32)
        exponent = ((bits >> 23) & 255) + (((bits >> 23) & 255) < 255).int() * ((bits & 0x7FFFFF) != 0).int()
    scale = torch.pow(2.0, exponent.float() - 127.0)
    normalized = rows / scale.unsqueeze(-1)
    magnitudes = torch.tensor(MXFP4[:8], device=values.device)
    distances = (normalized.abs().unsqueeze(-1) - magnitudes).abs()
    nearest = distances.argmin(-1)
    upper = (nearest + 1).clamp_max(7)
    ties = distances.gather(-1, nearest.unsqueeze(-1)) == distances.gather(-1, upper.unsqueeze(-1))
    nearest += ((nearest & 1) != 0).int() * ties.squeeze(-1).int() * (nearest != 7).int()
    return (torch.copysign(magnitudes[nearest], normalized) * scale.unsqueeze(-1)).flatten(1)


def exchange(tensor, send_splits, recv_splits):
    output = torch.empty((sum(recv_splits), *tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device)
    dist.all_to_all_single(
        output,
        tensor,
        output_split_sizes=recv_splits,
        input_split_sizes=send_splits,
    )
    return output


def make_routing(profile, rank, world, device):
    token = torch.arange(profile.tokens, dtype=torch.int32, device=device)[:, None]
    route = torch.arange(profile.topk, dtype=torch.int32, device=device)[None, :]
    local_experts = profile.experts // world
    destination = (rank + token + route) % world
    anchors = torch.tensor(
        sorted({value for value in (0, 1, 7, 8, 31, 63, 64, 95, local_experts - 1) if value < local_experts}),
        dtype=torch.int32,
        device=device,
    )
    anchor_index = (token + torch.div(route, world, rounding_mode="floor")) % anchors.numel()
    local_expert = anchors[anchor_index.long()]
    expert_ids = (destination * local_experts + local_expert).to(torch.int32).contiguous()
    unnormalized = 1.0 + ((token + 2 * route) % 7).float()
    expert_weights = (unnormalized / unnormalized.sum(1, keepdim=True)).contiguous()
    return expert_ids, expert_weights


def make_raw_weights(profile, local_experts, rank, device):
    generator = torch.Generator(device=device).manual_seed(0x4D4F45 + rank)

    def values(shape):
        return torch.randint(0, 256, (local_experts, *shape), generator=generator, dtype=torch.uint8, device=device)

    def scales(shape, low, high):
        return torch.randint(low, high, (local_experts, *shape), generator=generator, dtype=torch.uint8, device=device)

    raw_w13 = values((2 * profile.intermediate, profile.hidden // 2))
    raw_w2 = values((profile.hidden, profile.intermediate // 2))
    raw_s13 = scales(
        (2 * profile.intermediate, profile.hidden // 32),
        125 if profile.name == "kimi" else 118,
        129 if profile.name == "kimi" else 122,
    )
    raw_s2 = scales((profile.hidden, profile.intermediate // 32), 118, 122)
    raw_b1 = (
        torch.randn((local_experts, 2, profile.intermediate), generator=generator, dtype=torch.bfloat16, device=device)
        * 0.125
        if profile.bias
        else None
    )
    raw_b2 = (
        torch.randn((local_experts, profile.hidden), generator=generator, dtype=torch.bfloat16, device=device) * 0.125
        if profile.bias
        else None
    )
    return raw_w13, raw_w2, raw_s13, raw_s2, raw_b1, raw_b2


def reference(profile, x, expert_ids, expert_weights, raw, world):
    rank = dist.get_rank()
    local_experts = profile.experts // world
    raw_w13, raw_w2, raw_s13, raw_s2, raw_b1, raw_b2 = raw
    values = quantize_dequantize(x)
    routes = profile.tokens * profile.topk
    route_values = values[:, None, :].expand(-1, profile.topk, -1).reshape(routes, profile.hidden)
    route_ids = expert_ids.flatten()
    route_weights = expert_weights.flatten()
    destinations = torch.div(route_ids, local_experts, rounding_mode="floor")
    order = torch.argsort(destinations, stable=True)
    send_counts = torch.bincount(destinations, minlength=world).to(torch.int64)
    recv_counts = torch.empty_like(send_counts)
    dist.all_to_all_single(recv_counts, send_counts)
    send_splits = [int(value) for value in send_counts.cpu()]
    recv_splits = [int(value) for value in recv_counts.cpu()]

    received_values = exchange(route_values[order].contiguous(), send_splits, recv_splits)
    received_weights = exchange(route_weights[order, None].contiguous(), send_splits, recv_splits).flatten()
    local_ids = (route_ids % local_experts).to(torch.int64)
    received_experts = exchange(local_ids[order, None].contiguous(), send_splits, recv_splits).flatten()
    computed = torch.empty((received_values.shape[0], profile.hidden), dtype=torch.float32, device=x.device)

    for expert in torch.unique(received_experts).tolist():
        rows = torch.nonzero(received_experts == expert, as_tuple=False).flatten()
        w13 = dequantize(raw_w13[expert], raw_s13[expert]).transpose(0, 1)
        w2 = dequantize(raw_w2[expert], raw_s2[expert]).transpose(0, 1)
        hidden = received_values[rows] @ w13
        if raw_b1 is not None:
            hidden = hidden + raw_b1[expert].flatten().float()
        gate, up = hidden.split(profile.intermediate, dim=1)
        if profile.activation == "kimi_situ":
            activated = (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (25.0 * torch.tanh(up / 25.0))
        else:
            gate = gate.clamp_max(7.0)
            up = up.clamp(-7.0, 7.0) + 1.0
            activated = gate * torch.sigmoid(1.702 * gate) * up
        activated = quantize_dequantize(activated, intermediate=True)
        result = activated @ w2
        if raw_b2 is not None:
            result = result + raw_b2[expert].float()
        computed[rows] = (result * received_weights[rows, None]).to(torch.bfloat16).float()

    returned = exchange(computed, recv_splits, send_splits)
    route_output = torch.empty_like(returned)
    route_output[order] = returned
    expected = route_output.view(profile.tokens, profile.topk, profile.hidden).sum(1).to(torch.bfloat16)
    if not torch.isfinite(expected).all():
        raise AssertionError(f"rank {rank}: {profile.name} reference produced non-finite values")
    return expected


def check_output(profile, actual, expected):
    error = (actual.float() - expected).abs()
    scale = expected.square().mean().sqrt().clamp_min(1.0e-12)
    normalized = error / scale
    mean = normalized.mean().item()
    p99 = torch.quantile(normalized, 0.99).item()
    maximum = normalized.max().item()
    if not torch.isfinite(actual).all():
        raise AssertionError(f"{profile.name}: output contains non-finite values")
    if mean >= 0.04 or p99 >= 0.15 or maximum >= 0.30:
        raise AssertionError(f"{profile.name}: normalized error mean={mean:.4f}, p99={p99:.4f}, max={maximum:.4f}")


def run_profile(profile, gloo_group):
    rank, world = dist.get_rank(), dist.get_world_size()
    if profile.name == "kimi" and not any(solution.experts == profile.experts for solution in registered_solutions()):
        if rank == 0:
            print(f"{profile.name}: unsupported by this revision, skipped")
        return
    config = get_2stage_cfgs(
        profile.tokens,
        world,
        profile.experts,
        profile.topk,
        profile.hidden,
        profile.intermediate,
        activation=profile.activation,
        bias_dtype="bf16" if profile.bias else "none",
    )

    device = torch.device("cuda", torch.cuda.current_device())
    generator = torch.Generator(device=device).manual_seed(20260920 + rank)
    x = (torch.randn((profile.tokens, profile.hidden), generator=generator, device=device) * 0.125).to(torch.bfloat16)
    expert_ids, expert_weights = make_routing(profile, rank, world, device)
    raw = make_raw_weights(profile, config.local_experts, rank, device)
    expected = reference(profile, x, expert_ids, expert_weights, raw, world)
    weights = pack_expert_weights(config, *raw)
    workspace = MegaMoeWorkspace.allocate(config, group=gloo_group)
    actual = dynamic_mxfp4_mega_moe(x, weights, expert_ids, expert_weights, workspace=workspace)
    torch.cuda.synchronize()
    check_output(profile, actual, expected)
    dist.barrier(group=gloo_group)
    if rank == 0:
        print(f"{profile.name}: MegaMoE output matches the distributed reference")


def main():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", device_id=torch.cuda.current_device())
    gloo_group = dist.new_group(backend="gloo")
    try:
        if dist.get_world_size() != 8:
            raise RuntimeError("MegaMoE correctness test requires 8 ranks")
        for profile in PROFILES:
            run_profile(profile, gloo_group)
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
