"""Graph-safe, intra-node dynamic MXFP4 MegaMoE API."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..local_moe.api import ExpertWeights
from .config import MegaMoeConfig
from .dispatch import get_2stage_cfgs, resolve_implementation
from .quantization import quantize_input
from .solutionid import DataType
from .symmetric_heap import SymmetricHeap
from .workspace import WorkspaceLayout


@dataclass(frozen=True)
class MegaMoeInputViews:
    act: torch.Tensor
    scales: torch.Tensor
    expert_ids: torch.Tensor
    expert_weights: torch.Tensor


def pack_expert_weights(config, w13, w2, s13, s2, bias1=None, bias2=None):
    """Pad logical hidden columns/rows, then use the shared native weight codec.

    Inputs are local experts' row-major E2M1 bytes and E8M0 scales; biases are
    BF16. The returned ExpertWeights also works with the padded local MoE path.
    """
    e, d, i = config.local_experts, config.solution.hidden, config.solution.intermediate
    for value, shape in (
        (w13, (e, 2 * i, d // 2)),
        (w2, (e, d, i // 2)),
        (s13, (e, 2 * i, d // 32)),
        (s2, (e, d, i // 32)),
    ):
        if value.shape != shape or value.dtype != torch.uint8 or value.device != w13.device:
            raise ValueError("raw weights/scales must have the configured local expert and logical hidden dimensions")
    padding = config.compute_hidden - d
    if padding:
        w13 = F.pad(w13, (0, padding // 2))
        s13 = F.pad(s13, (0, padding // 32), value=127)
        w2 = F.pad(w2, (0, 0, 0, padding))
        s2 = F.pad(s2, (0, 0, 0, padding), value=127)
        if bias2 is not None:
            bias2 = F.pad(bias2, (0, padding))
    return ExpertWeights.pack(w13.contiguous(), w2.contiguous(), s13.contiguous(), s2.contiguous(), bias1, bias2)


@dataclass
class MegaMoeWorkspace:
    config: MegaMoeConfig
    heap: SymmetricHeap
    memory: torch.Tensor
    inputs: MegaMoeInputViews
    output: torch.Tensor

    @classmethod
    def allocate(cls, config, *, group=None):
        resolve_implementation(config)
        if not torch.cuda.is_available() or not torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).gcnArchName.startswith("gfx950"):
            raise ValueError("MegaMoE requires gfx950 GPUs")
        layout = WorkspaceLayout(config)
        heap = SymmetricHeap(layout, world_size=config.solution.world_size, group=group)
        cap, topk, d, stride = (
            config.max_tokens_per_rank,
            config.solution.topk,
            config.solution.hidden,
            config.input_token_bytes,
        )
        base = layout.rank_base(heap.rank)
        rows = heap.tensor(base + layout.input_tokens, cap * stride).view(cap, stride)
        inputs = MegaMoeInputViews(
            rows[:, : d // 2],
            rows[:, d // 2 : d // 2 + d // 32],
            heap.tensor(layout.input_ids, cap * topk * 4).view(torch.int32).view(cap, topk),
            heap.tensor(base + layout.input_weights, cap * topk * 4).view(torch.float32).view(cap, topk),
        )
        return cls(
            config,
            heap,
            heap.tensor(),
            inputs,
            torch.empty((cap, d), dtype=torch.bfloat16, device=f"cuda:{heap.device}"),
        )

    def input_views(self, tokens):
        self.config.validate_tokens(tokens)
        return MegaMoeInputViews(
            self.inputs.act[:tokens],
            self.inputs.scales[:tokens],
            self.inputs.expert_ids[:tokens],
            self.inputs.expert_weights[:tokens],
        )

    def quantize(self, x):
        if x.ndim != 2 or x.shape[1] != self.config.solution.hidden or x.device != self.output.device:
            raise ValueError("input must match the workspace hidden size and GPU")
        self.config.validate_tokens(x.shape[0])
        if torch.cuda.current_device() != self.heap.device:
            raise ValueError("select the workspace GPU before quantization")
        views = self.input_views(x.shape[0])
        return quantize_input(x, out=(views.act, views.scales))

    def run(self, weights, tokens, *, inputs=None, out=None):
        """Run quantized inputs; every rank participates, including zero-token ranks.

        Every token must select `topk` distinct global expert IDs in [0, E).
        The heap is shared by sequential invocations and graphs on one stream;
        concurrent invocations require separate workspaces/process groups.
        """
        solution = self.config.solution
        config = get_2stage_cfgs(
            tokens,
            solution.world_size,
            solution.experts,
            solution.topk,
            solution.hidden,
            solution.intermediate,
            activation=solution.activation,
            bias_dtype=solution.bias_dtype,
        )
        factory = resolve_implementation(config)
        if torch.cuda.current_device() != self.heap.device:
            raise ValueError("select the workspace GPU before running MegaMoE")
        inputs = self.input_views(tokens) if inputs is None else inputs
        d, i, e, topk = config.compute_hidden, config.solution.intermediate, config.local_experts, config.solution.topk
        device = self.output.device
        for value, shape, dtype in (
            (weights.w13, (e, 2 * i, d // 2), torch.uint8),
            (weights.w2, (e, d, i // 2), torch.uint8),
            (weights.s13, (e, 2 * i, d // 32), torch.uint8),
            (weights.s2, (e, d, i // 32), torch.uint8),
            (inputs.expert_ids, (tokens, topk), torch.int32),
            (inputs.expert_weights, (tokens, topk), torch.float32),
        ):
            if value.shape != shape or value.dtype != dtype or value.device != device or not value.is_contiguous():
                raise ValueError(
                    "weights/routing must match the configured shape, dtype and GPU, with contiguous storage"
                )
        for value, shape in ((weights.bias1, (e, 2, i)), (weights.bias2, (e, d))):
            if value is not None and (
                config.solution.bias_dtype != DataType.BF16
                or value.shape != shape
                or value.dtype != torch.bfloat16
                or value.device != device
                or not value.is_contiguous()
            ):
                raise ValueError("bias must be packed BF16 with the configured shape and GPU")
        if (
            inputs.act.shape != (tokens, config.solution.hidden // 2)
            or inputs.act.dtype != torch.uint8
            or inputs.act.device != device
            or inputs.act.stride() != (config.input_token_bytes, 1)
            or inputs.scales.shape != (tokens, config.solution.hidden // 32)
            or inputs.scales.dtype != torch.uint8
            or inputs.scales.device != device
            or inputs.scales.stride() != (config.input_token_bytes, 1)
            or (tokens and inputs.scales.data_ptr() != inputs.act.data_ptr() + config.solution.hidden // 2)
        ):
            raise ValueError("input act and scales must share the configured packed rows")
        out = self.output[:tokens] if out is None else out
        if (
            out.shape != (tokens, config.solution.hidden)
            or out.dtype != torch.bfloat16
            or out.device != device
            or out.stride(1) != 1
            or out.stride(0) not in (config.solution.hidden, config.compute_hidden)
        ):
            raise ValueError("out must be BF16 [tokens, hidden] with a logical or padded row stride")
        count, plan, push, stage1, stage2, combine, barrier = factory(config)
        rank = self.heap.rank
        count[lambda: ((1, 1, 1), (256, 1, 1))](self.memory, inputs.expert_ids, tokens, rank)
        barrier[lambda: ((1, 1, 1), (64, 1, 1))](self.memory, rank, num_warps=1)
        plan[lambda: ((1, 1, 1), (256, 1, 1))](self.memory, rank)
        barrier[lambda: ((1, 1, 1), (64, 1, 1))](self.memory, rank, num_warps=1)
        push[lambda: ((config.solution.experts, 1, 1), (256, 1, 1))](
            self.memory, inputs.act, inputs.expert_weights, tokens, rank
        )
        barrier[lambda: ((1, 1, 1), (64, 1, 1))](self.memory, rank, num_warps=1)
        stage1[lambda: ((256, 1, 1), (256, 1, 1))](
            self.memory, weights.w13, weights.s13, weights.bias1, rank, int(weights.bias1 is not None)
        )
        stage2[lambda: ((256, 1, 1), (256, 1, 1))](
            self.memory, weights.w2, weights.s2, weights.bias2, rank, int(weights.bias2 is not None)
        )
        barrier[lambda: ((1, 1, 1), (64, 1, 1))](self.memory, rank, num_warps=1)
        combine[lambda: ((config.max_tokens_per_rank, 1, 1), (256, 1, 1))](
            self.memory, out, tokens, out.stride(0), rank
        )
        barrier[lambda: ((1, 1, 1), (64, 1, 1))](self.memory, rank, num_warps=1)
        return out


def dynamic_mxfp4_mega_moe(x, weights, expert_ids, expert_weights, *, workspace, out=None):
    """Quantize BF16 inputs, dispatch global routes, compute and combine locally."""
    act, scales = workspace.quantize(x)
    return workspace.run(
        weights, x.shape[0], inputs=MegaMoeInputViews(act, scales, expert_ids, expert_weights), out=out
    )
