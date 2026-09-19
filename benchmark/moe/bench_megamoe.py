"""Benchmark Avelang or AITER Kimi K3 MegaMoE full paths.

This follows Petit's MegaMoE benchmark: every timed full-path iteration starts
from a disjoint per-rank BF16 latent batch, computes normalized top-k routing,
then runs Avelang direct push or vLLM's all-gather/fused_moe/reduce-scatter
AITER path. Both backends use the same deterministic input, routing, and MXFP4
weight generation so separately saved outputs can be compared outside this benchmark.

Only Kimi EP8 is covered here.  Kimi TP is intentionally outside this port.
"""

import argparse
import gc
import json
import math
import os
import statistics
import sys
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch
import torch.distributed as dist
from avelang_kernels.amdgpu.mega_moe import MegaMoeWorkspace, get_2stage_cfgs, pack_expert_weights

MODELS = {
    "gptoss": (32, 4, 2880, 3072, "swiglu", "bf16"),
    "gptoss120b": (128, 4, 2880, 3072, "swiglu", "bf16"),
    "dsv32": (256, 8, 7168, 2048, "silu", "none"),
    "dsv4": (384, 6, 7168, 3072, "silu", "none"),
    "kimi": (896, 16, 3584, 3072, "kimi_situ", "none"),
}


@dataclass
class Backend:
    name: str
    prepare: Callable[[], None]
    compute: Callable[[], torch.Tensor]


def measure(fn, args, nccl_group):
    alignment = torch.zeros(1, dtype=torch.int32, device="cuda") if args.device_align_graph else None
    # Prime shape-specific JIT work and the NCCL communicator before graph
    # capture.  NCCL cannot lazily initialize a communicator while capturing.
    for _ in range(args.warmup):
        if args.device_align_graph:
            dist.all_reduce(alignment, group=nccl_group)
            torch.cuda.synchronize()
        fn()
        if args.device_align_graph:
            torch.cuda.synchronize()
    torch.cuda.synchronize()
    dist.barrier()

    graph = torch.cuda.CUDAGraph()
    capture_stream = torch.cuda.Stream()
    if args.device_align_graph:
        timed_start = torch.cuda.Event(enable_timing=True, external=True)
        timed_end = torch.cuda.Event(enable_timing=True, external=True)
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        # Match Petit: warm the exact capture stream before capture. This also
        # registers collective resources with the graph-capable communicator.
        for _ in range(3):
            if args.device_align_graph:
                dist.all_reduce(alignment, group=nccl_group)
            fn()
        if not args.device_align_graph:
            with torch.cuda.graph(graph):
                for _ in range(args.graph_iters):
                    fn()
    if args.device_align_graph:
        capture_stream.synchronize()
        dist.barrier()
        with torch.cuda.stream(capture_stream), torch.cuda.graph(graph, stream=capture_stream):
            dist.all_reduce(alignment, group=nccl_group)
            timed_start.record()
            for _ in range(args.graph_iters):
                fn()
            timed_end.record()
    torch.cuda.current_stream().wait_stream(capture_stream)

    # The first replay may upload kernels and register collective resources.
    graph_warmup_replays = (
        math.ceil(args.warmup / args.graph_iters) if args.graph_warmup_replays is None else args.graph_warmup_replays
    )
    for _ in range(graph_warmup_replays):
        graph.replay()
    torch.cuda.synchronize()
    dist.barrier()

    gc_frozen = False
    try:
        if args.freeze_gc_during_timing:
            gc.collect()
            gc.freeze()
            gc_frozen = True
            dist.barrier()
        local_samples = []
        for _ in range(args.repeat):
            dist.barrier()
            if args.device_align_graph:
                graph.replay()
                timed_end.synchronize()
                elapsed = timed_start.elapsed_time(timed_end) * 1000 / args.graph_iters
            else:
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                graph.replay()
                end.record()
                end.synchronize()
                elapsed = start.elapsed_time(end) * 1000 / args.graph_iters
            local_samples.append(elapsed)
        local_times = torch.tensor(local_samples, dtype=torch.float32, device="cuda")
        gathered_times = [torch.empty_like(local_times) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_times, local_times, group=nccl_group)
        samples = torch.stack(gathered_times).amax(dim=0).cpu().tolist()
    finally:
        if gc_frozen:
            gc.unfreeze()
    return {
        "median_us": statistics.median(samples),
        "min_us": min(samples),
        "max_us": max(samples),
    }


@lru_cache(maxsize=8)
def build_kimi_raw_weights(rank, seed, device_index):
    """Generate Petit's EP8 Kimi weights from TP-aligned BF16 blocks."""
    from aiter.ops.triton.moe.quant_moe import downcast_to_mxfp

    device = torch.device("cuda", device_index)
    experts, dim, inter, shard_size = 112, 3584, 3072, 384
    w1 = torch.empty((experts, 2 * inter, dim // 2), dtype=torch.uint8, device=device)
    w2 = torch.empty((experts, dim, inter // 2), dtype=torch.uint8, device=device)
    s1 = torch.empty((experts, 2 * inter, dim // 32), dtype=torch.uint8, device=device)
    s2 = torch.empty((experts, dim, inter // 32), dtype=torch.uint8, device=device)
    generator = torch.Generator(device=device)
    for expert in range(experts):
        global_expert = rank * experts + expert
        for shard in range(8):
            generator.manual_seed(seed + 1_000_003 * global_expert + 97 * shard)
            row = shard * shard_size
            for projection in range(3):
                shape = (shard_size, dim) if projection < 2 else (dim, shard_size)
                bf16 = (torch.randn(shape, generator=generator, device=device) * 40).to(torch.bfloat16)
                value, scale = downcast_to_mxfp(bf16, torch.uint8, axis=-1)
                if projection < 2:
                    begin = projection * inter + row
                    w1[expert, begin : begin + shard_size].copy_(value)
                    s1[expert, begin : begin + shard_size].copy_(scale.view(torch.uint8))
                else:
                    w2[expert, :, row // 2 : (row + shard_size) // 2].copy_(value)
                    s2[expert, :, row // 32 : (row + shard_size) // 32].copy_(scale.view(torch.uint8))
    return w1, w2, s1, s2


def build_random_weights(config, seed):
    torch.manual_seed(seed + dist.get_rank())
    local, d, i = config.local_experts, config.solution.hidden, config.solution.intermediate
    return (
        torch.randint(0, 256, (local, 2 * i, d // 2), dtype=torch.uint8, device="cuda"),
        torch.randint(0, 256, (local, d, i // 2), dtype=torch.uint8, device="cuda"),
        torch.full((local, 2 * i, d // 32), 119, dtype=torch.uint8, device="cuda"),
        torch.full((local, d, i // 32), 119, dtype=torch.uint8, device="cuda"),
    )


def make_avelang_backend(config, x, expert_ids, expert_weights, raw, gloo_group):
    rank = dist.get_rank()
    e, _, _, _, _, bias = (
        config.solution.experts,
        config.solution.topk,
        config.solution.hidden,
        config.solution.intermediate,
        config.solution.activation,
        config.solution.bias_dtype,
    )
    w1, w2, s1, s2 = raw
    local, d, i = config.local_experts, config.compute_hidden, config.solution.intermediate
    b1 = torch.zeros((local, 2, i), dtype=torch.bfloat16, device="cuda") if int(bias) else None
    b2 = torch.zeros((local, d), dtype=torch.bfloat16, device="cuda") if int(bias) else None
    weights = pack_expert_weights(config, w1, w2, s1, s2, b1, b2)
    workspace = MegaMoeWorkspace.allocate(config, group=gloo_group)
    inputs = workspace.input_views(x.shape[0])

    def prepare():
        inputs.expert_ids.copy_(expert_ids)
        inputs.expert_weights.copy_(expert_weights)
        workspace.quantize(x)

    def compute():
        return workspace.run(weights, x.shape[0])

    prepare()
    if rank == 0:
        print(f"allocated Avelang E{e} workspace", file=sys.stderr)
    return Backend("avelang", prepare, compute)


class AiterEPDispatcher:
    def __init__(self, m, topk, hidden, group):
        world = dist.get_world_size(group)
        self.group = group
        self.hidden = torch.empty((world * m, hidden), dtype=torch.bfloat16, device="cuda")
        self.weights = torch.empty((world * m, topk), dtype=torch.float32, device="cuda")
        self.ids = torch.empty((world * m, topk), dtype=torch.int32, device="cuda")
        self.output = torch.empty((m, hidden), dtype=torch.bfloat16, device="cuda")

    def dispatch(self, hidden, ids, weights):
        dist.all_gather_into_tensor(self.hidden, hidden.contiguous(), group=self.group)
        dist.all_gather_into_tensor(self.ids, ids.contiguous(), group=self.group)
        dist.all_gather_into_tensor(self.weights, weights.contiguous(), group=self.group)

    def combine(self, local):
        dist.reduce_scatter_tensor(self.output, local.contiguous(), op=dist.ReduceOp.SUM, group=self.group)
        return self.output


def make_aiter_backend(config, x, expert_ids, expert_weights, raw, nccl_group):
    from aiter import ActivationType, QuantType, dtypes
    from aiter.fused_moe import fused_moe
    from aiter.ops.shuffle import shuffle_weight
    from aiter.utility.fp4_utils import e8m0_shuffle

    rank = dist.get_rank()
    local, d, i = config.local_experts, config.solution.hidden, config.solution.intermediate
    w1, w2, s1, s2 = raw
    os.environ["AITER_SITUV2_A4W4"] = "1"
    os.environ["AITER_SITUV2_A8W4"] = "0"
    w1 = shuffle_weight(w1.view(dtypes.fp4x2), (16, 16))
    w2 = shuffle_weight(w2.view(dtypes.fp4x2), (16, 16))
    s1 = e8m0_shuffle(s1.view(local * 2 * i, d // 32)).view(dtypes.fp8_e8m0)
    s2 = e8m0_shuffle(s2.view(local * d, i // 32)).view(dtypes.fp8_e8m0)
    dispatcher = AiterEPDispatcher(x.shape[0], config.solution.topk, d, nccl_group)
    mask = torch.zeros(config.solution.experts, dtype=torch.int32, device="cuda")
    mask[rank * local : (rank + 1) * local] = 1
    state = {}

    def prepare():
        dispatcher.dispatch(x, expert_ids, expert_weights)

    def expert_compute():
        state["local"] = fused_moe(
            dispatcher.hidden,
            w1,
            w2,
            dispatcher.weights,
            dispatcher.ids,
            expert_mask=mask,
            activation=ActivationType.Situv2,
            quant_type=QuantType.per_1x32,
            w1_scale=s1,
            w2_scale=s2,
            dtype=torch.bfloat16,
            hidden_pad=0,
            intermediate_pad=0,
            beta=4.0,
            linear_beta=25.0,
            gate_mode="separated",
        )
        return state["local"]

    def compute():
        expert_compute()
        return dispatcher.combine(state["local"])

    prepare()
    if rank == 0:
        print("using AITER Kimi a4w4", file=sys.stderr)
    return Backend("aiter-a4w4", prepare, compute)


def stage_topk(x, source_logits, expert_ids, expert_weights, topk):
    """Use the same normalized fused top-k operation as Petit's benchmark."""
    from aiter.fused_moe import fused_topk

    fused_topk(
        x,
        source_logits,
        topk,
        True,
        topk_ids=expert_ids,
        topk_weights=expert_weights,
    )


def save_output(args, output, expert_ids, expert_weights, global_tokens, gloo_group):
    if args.save_output_dir is None:
        return
    payload = {
        "output": output.detach().cpu(),
        "topk_ids": expert_ids.detach().cpu(),
        "topk_weights": expert_weights.detach().cpu(),
    }
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, payload, group=gloo_group)
    if dist.get_rank() == 0:
        global_payload = {name: torch.cat([part[name] for part in gathered]) for name in payload}
        args.save_output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            global_payload,
            args.save_output_dir / f"{args.backend}_ep_m{global_tokens}.pt",
        )


def run_case(args, model, tokens, nccl_group, gloo_group):
    rank, world = dist.get_rank(), dist.get_world_size()
    e, topk, d, i, act, bias = MODELS[model]
    if model == "kimi" and world != 8:
        raise ValueError("Kimi MegaMoE benchmark requires EP8")
    if args.backend == "aiter" and model != "kimi":
        raise ValueError("the AITER comparison in this benchmark is Kimi EP only")
    config = get_2stage_cfgs(tokens, world, e, topk, d, i, activation=act, bias_dtype=bias)
    # Match Petit: build one logical global batch deterministically on every
    # rank, then give each EP rank a disjoint row slice.
    global_tokens = tokens * world
    torch.manual_seed(args.seed + global_tokens)
    global_x = torch.randn((global_tokens, d), dtype=torch.float32, device="cuda").to(torch.bfloat16)
    source_logits = torch.randn((global_tokens, e), dtype=torch.float32, device="cuda")
    begin = rank * tokens
    x = global_x.narrow(0, begin, tokens).contiguous()
    source_logits = source_logits.narrow(0, begin, tokens).contiguous()
    expert_ids = torch.empty((tokens, topk), dtype=torch.int32, device="cuda")
    expert_weights = torch.empty((tokens, topk), dtype=torch.float32, device="cuda")
    stage_topk(x, source_logits, expert_ids, expert_weights, topk)
    raw = (
        build_kimi_raw_weights(rank, args.seed, torch.cuda.current_device())
        if model == "kimi"
        else build_random_weights(config, args.seed)
    )

    if args.backend == "avelang":
        backend = make_avelang_backend(config, x, expert_ids, expert_weights, raw, gloo_group)
    else:
        backend = make_aiter_backend(config, x, expert_ids, expert_weights, raw, nccl_group)
    torch.cuda.synchronize()

    def full_path():
        stage_topk(x, source_logits, expert_ids, expert_weights, topk)
        backend.prepare()
        return backend.compute()

    if args.save_output_dir is not None:
        output = full_path().detach().clone()
        torch.cuda.synchronize()
        save_output(args, output, expert_ids, expert_weights, global_tokens, gloo_group)

    timings = {}
    for scope in args.scopes:
        fn = backend.compute if scope == "compute" else full_path
        timings[scope] = measure(fn, args, nccl_group)
        if rank == 0:
            value = timings[scope]["median_us"]
            print(
                f"{args.backend} {model} EP{world} M{tokens} {scope}: {value:.2f} us",
                file=sys.stderr,
            )
    dist.barrier()
    return {
        "model": model,
        "tokens_per_rank": tokens,
        "global_tokens": global_tokens,
        "solution_id": int(config.solution),
        "backend_mode": backend.name,
        "timings": timings,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=["kimi"])
    parser.add_argument("--tokens", nargs="+", type=int, default=[8, 32, 128])
    parser.add_argument("--backend", choices=["avelang", "aiter"], default="avelang")
    parser.add_argument("--scopes", nargs="+", choices=["compute", "full"], default=["full"])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument(
        "--graph-iters",
        type=int,
        default=1,
        help="For Kimi use 1, matching Petit's driver for distributed full paths.",
    )
    parser.add_argument(
        "--graph-warmup-replays",
        type=int,
        help="Captured-graph warmups; defaults to ceil(warmup / graph-iters).",
    )
    parser.add_argument(
        "--freeze-gc-during-timing",
        action="store_true",
        help="Freeze the initialized Python heap during graph replay timing.",
    )
    parser.add_argument(
        "--device-align-graph",
        action="store_true",
        help="Align ranks inside the captured graph and exclude that wait from timing.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save-output-dir",
        type=Path,
        help="Optionally save the global output and routing for external checks.",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON output; defaults to rank-zero stdout")
    args = parser.parse_args()
    if (
        args.warmup < 1
        or args.repeat < 1
        or args.graph_iters < 1
        or (args.graph_warmup_replays is not None and args.graph_warmup_replays < 0)
        or any(not 0 < m <= 1024 for m in args.tokens)
    ):
        parser.error("timing values must be positive and tokens must be in [1,1024]")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    # Match Petit's benchmark: NCCL is the default timing/collective group.
    # Avelang's VMM handle exchange uses a separate CPU/Gloo group.
    dist.init_process_group("nccl", device_id=torch.cuda.current_device())
    nccl_group = dist.group.WORLD
    gloo_group = dist.new_group(backend="gloo")
    try:
        report = {
            "backend": args.backend,
            "world_size": dist.get_world_size(),
            "input": "Petit's deterministic global BF16 latent batch and normalized fused top-k routing",
            "gpu": torch.cuda.get_device_name(),
            "seed": args.seed,
            "timed_boundary": "BF16 latent input through top-k, prepare/dispatch, expert compute and combine",
            "aiter_ep": "all-gather + public fused_moe + reduce-scatter",
            "timing": {
                "warmup": args.warmup,
                "repeat": args.repeat,
                "graph_iters": args.graph_iters,
                "graph_warmup_replays": args.graph_warmup_replays,
                "freeze_gc_during_timing": args.freeze_gc_during_timing,
                "device_align_graph": args.device_align_graph,
                "statistic": "median of per-sample maximum rank latency",
            },
            "cases": [run_case(args, model, m, nccl_group, gloo_group) for model in args.models for m in args.tokens],
        }
        if dist.get_rank() == 0:
            encoded = json.dumps(report, indent=2) + "\n"
            if args.output:
                path = args.output.expanduser()
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(encoded)
            else:
                print(encoded, end="")
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
