"""Dynamic MXFP4 two-stage MoE with caller-owned, graph-safe workspace."""

from dataclasses import dataclass, replace

import torch

from .config import MoeConfig
from .dispatch import resolve_2stage_implementation
from .intermediate_mxfp4 import IntermediateLayout
from .scale_layout import scale_byte_shape
from .solutionid import DataType
from .stage1 import make_stage1
from .stage2 import make_stage2


@dataclass(frozen=True)
class Routing:
    """AITER-style expert-sorted routes, with padding at each tile's end."""

    ids: torch.Tensor
    weights: torch.Tensor
    experts: torch.Tensor
    counts: torch.Tensor

    @property
    def capacity(self):
        return self.ids.numel()


def _pack_weights(values: torch.Tensor, scales: torch.Tensor):
    """Pack E2M1 bytes [E,N,K/2] and E8M0 scales [E,N,K/32]."""
    if values.ndim != 3 or values.dtype != torch.uint8 or not values.is_contiguous():
        raise ValueError("weights must be contiguous uint8 [experts, N, K/2]")
    experts, n, half_k = values.shape
    k = 2 * half_k
    if n % 256 or k % 256:
        raise ValueError("native MoE weights require N and K divisible by 256")
    if (
        scales.shape != (experts, n, k // 32)
        or scales.dtype != torch.uint8
        or scales.device != values.device
        or not scales.is_contiguous()
    ):
        raise ValueError("weight scales must be contiguous uint8 [experts, N, K/32] on the same device")
    words = values.view(torch.int32).reshape(experts, n // 16, 16, k // 128, 4, 4)
    # [expert, N16 tile, K128 tile, K32 lane group, N lane, register]
    words = words.permute(0, 1, 3, 4, 2, 5).contiguous().view(experts, n // 16, k // 128, 4, 16, 4)
    tiled_scales = scales.reshape(experts, n // 32, 2, 16, (k // 256), 2, 4)
    tiled_scales = tiled_scales.permute(0, 1, 4, 6, 3, 5, 2).contiguous()
    return words.view(torch.uint8).reshape_as(values), tiled_scales.reshape_as(scales)


def _pack_bias(bias: torch.Tensor) -> torch.Tensor:
    """Preserve the Petit bias ordering used with weight-as-A MFMA."""
    if bias.dtype != torch.bfloat16 or bias.ndim not in (2, 3) or not bias.is_contiguous():
        raise ValueError("bias must be contiguous BF16 [E,N] or [E,2,N]")
    if bias.shape[-1] % 256:
        raise ValueError("bias columns must be divisible by 256")
    tiles = bias.reshape(-1, bias.shape[-1] // 256, 4, 4, 4, 4)
    return tiles.permute(0, 1, 2, 4, 3, 5).contiguous().reshape_as(bias)


@dataclass(frozen=True)
class ExpertWeights:
    w13: torch.Tensor
    w2: torch.Tensor
    s13: torch.Tensor
    s2: torch.Tensor
    bias1: torch.Tensor | None = None
    bias2: torch.Tensor | None = None

    @classmethod
    def pack(cls, w13, w2, s13, s2, bias1=None, bias2=None):
        w13, s13 = _pack_weights(w13, s13)
        w2, s2 = _pack_weights(w2, s2)
        return cls(
            w13,
            w2,
            s13,
            s2,
            _pack_bias(bias1) if bias1 is not None else None,
            _pack_bias(bias2) if bias2 is not None else None,
        )


@dataclass
class MoeWorkspace:
    input_act: torch.Tensor
    input_scales: torch.Tensor
    sorted_scales: torch.Tensor
    intermediate: torch.Tensor
    out: torch.Tensor

    @classmethod
    def allocate(cls, x: torch.Tensor, routing: Routing, config: MoeConfig):
        m, d = x.shape
        layout = IntermediateLayout(routing.capacity, config.intermediate)

        def byte_tensor(shape):
            return torch.empty(shape, dtype=torch.uint8, device=x.device)

        return cls(
            byte_tensor((m, d // 2)),
            byte_tensor((m, d // 32)),
            byte_tensor(scale_byte_shape(routing.capacity, d)),
            byte_tensor((layout.nbytes,)),
            torch.empty_like(x),
        )


def prepare_input(x: torch.Tensor, routing: Routing, config: MoeConfig, workspace: MoeWorkspace):
    """Run AITER input quantization/sorting into caller-owned buffers."""
    tokens, columns = x.shape
    act, per_token, scales = workspace.input_act, workspace.input_scales, workspace.sorted_scales
    for tensor, shape in (
        (act, (tokens, columns // 2)),
        (per_token, (tokens, columns // 32)),
        (scales, scale_byte_shape(routing.capacity, columns)),
    ):
        if (
            tensor.shape != shape
            or tensor.dtype != torch.uint8
            or tensor.device != x.device
            or not tensor.is_contiguous()
        ):
            raise ValueError("input-preparation workspace has incompatible shape, dtype, device or strides")
    if not tokens:
        return act, scales

    # Keep AITER a dependency of BF16 input preparation, not kernel imports.
    from aiter.ops.quant import (
        dynamic_per_group_scaled_quant,
        fused_dynamic_mx_quant_moe_sort_hip,
        mxfp4_moe_sort_hip,
    )

    sorted_scales = scales.view(routing.capacity, columns // 32)
    if tokens <= 2048 // config.topk:
        fused_dynamic_mx_quant_moe_sort_hip(
            act, sorted_scales, x, routing.ids, routing.counts, tokens, config.stage1_tile_m, 32
        )
    else:
        dynamic_per_group_scaled_quant(act, x, per_token, 32, shuffle_scale=False)
        mxfp4_moe_sort_hip(sorted_scales, per_token, routing.ids, routing.counts, tokens, columns)
    return act, scales


def dynamic_mxfp4_moe(
    x: torch.Tensor,
    weights: ExpertWeights,
    routing: Routing,
    config: MoeConfig,
    *,
    workspace: MoeWorkspace | None = None,
):
    """Quantize BF16 input, compute both expert projections, return BF16 output.

    Route IDs are expert-sorted with trailing padding in each config.stage1_tile_m tile, as
    produced by AITER moe_sorting. Counts stay on the GPU
    and contain [padded route extent, token count]. Prepare weights and routing
    outside this call; pass a preallocated workspace for repeated/graph use.
    """
    resolve_2stage_implementation(config)
    if x.ndim != 2 or x.shape[1] != config.hidden or x.dtype != torch.bfloat16:
        raise ValueError("input must be BF16 [tokens, hidden]")
    if x.device.type != "cuda" or not x.is_contiguous():
        raise ValueError("input must be a contiguous GPU tensor")
    if torch.cuda.get_device_properties(x.device).gcnArchName.split(":")[0] != "gfx950":
        raise ValueError("dynamic MXFP4 MoE requires gfx950")
    capacity = routing.capacity
    if capacity % config.stage1_tile_m or capacity < x.shape[0] * config.topk:
        raise ValueError("routing capacity must cover all routes and be divisible by the Stage1 tile M")
    for tensor, dtype, count in (
        (routing.ids, torch.int32, capacity),
        (routing.weights, torch.float32, capacity),
        (routing.experts, torch.int32, capacity // config.stage1_tile_m),
        (routing.counts, torch.int32, 2),
    ):
        if tensor.dtype != dtype or tensor.numel() != count or tensor.device != x.device or not tensor.is_contiguous():
            raise ValueError("routing tensor has incompatible dtype, size, device or strides")
    e, d, i = config.experts, config.hidden, config.intermediate
    for tensor, shape in (
        (weights.w13, (e, 2 * i, d // 2)),
        (weights.w2, (e, d, i // 2)),
        (weights.s13, (e, 2 * i, d // 32)),
        (weights.s2, (e, d, config.scale_columns)),
    ):
        if (
            tensor.shape != shape
            or tensor.dtype != torch.uint8
            or tensor.device != x.device
            or not tensor.is_contiguous()
        ):
            raise ValueError("expert weights/scales have incompatible shape, dtype, device or strides")
    if config.bias:
        for tensor, shape in ((weights.bias1, (e, 2, i)), (weights.bias2, (e, d))):
            if tensor is None:
                continue
            if (
                tensor.shape != shape
                or tensor.dtype != torch.bfloat16
                or tensor.device != x.device
                or not tensor.is_contiguous()
            ):
                raise ValueError("bias must be packed BF16 [E,2,I] and [E,D]")
    elif weights.bias1 is not None or weights.bias2 is not None:
        raise ValueError("bias tensors require solution.bias_dtype=BF16")
    if workspace is None:
        workspace = MoeWorkspace.allocate(x, routing, config)
    if (
        workspace.out.shape != x.shape
        or workspace.out.device != x.device
        or workspace.out.dtype != x.dtype
        or not workspace.out.is_contiguous()
        or workspace.intermediate.device != x.device
    ):
        raise ValueError("workspace output must match input shape, device and dtype")
    IntermediateLayout(capacity, i).views(workspace.intermediate, x.shape[0], config.topk)
    out = workspace.out
    if not x.shape[0]:
        return out
    with torch.cuda.device(x.device):
        _, input_scales = prepare_input(x, routing, config, workspace)
        no_bias = config
        if config.bias and (weights.bias1 is None or weights.bias2 is None):
            no_bias = replace(config, solution=replace(config.solution, bias_dtype=DataType.NONE))
        config1 = no_bias if weights.bias1 is None else config
        config2 = no_bias if weights.bias2 is None else config
        bias1 = weights.bias1 if config1.bias else out
        bias2 = weights.bias2 if config2.bias else out
        make_stage1(config1)[lambda: config1.stage1_grid(capacity)](
            workspace.input_act,
            input_scales,
            weights.w13,
            weights.s13,
            bias1,
            routing.ids,
            routing.experts,
            routing.counts,
            workspace.intermediate,
            capacity,
            num_warps=config1.stage1_num_warps,
        )
        out.zero_()
        make_stage2(config2)[lambda: config2.stage2_grid()](
            workspace.intermediate,
            weights.w2,
            weights.s2,
            bias2,
            routing.ids,
            routing.experts,
            routing.weights,
            routing.counts,
            out,
            capacity,
            num_warps=config2.stage2_num_warps,
        )
        return out
