"""Benchmark AveLang MegaMoE with synthetic inputs; launch with torchrun."""

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from avelang_kernels.amdgpu.mega_moe import MegaMoeWorkspace, get_2stage_cfgs, pack_expert_weights

MODELS = {
    "gptoss": (32, 4, 2880, 3072, "swiglu", "bf16"),
    "gptoss120b": (128, 4, 2880, 3072, "swiglu", "bf16"),
    "dsv32": (256, 8, 7168, 2048, "silu", "none"),
    "dsv4": (384, 6, 7168, 3072, "silu", "none"),
}


def measure(fn, args):
    for _ in range(args.warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(args.graph_iters):
            fn()
    samples = []
    for _ in range(args.repeat):
        dist.barrier()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        elapsed = start.elapsed_time(end) * 1000 / args.graph_iters
        per_rank = [None] * dist.get_world_size()
        dist.all_gather_object(per_rank, elapsed)
        samples.append(max(per_rank))
    return {"median_us": statistics.median(samples), "min_us": min(samples), "max_us": max(samples)}


def run_case(args, model, tokens):
    rank, world = dist.get_rank(), dist.get_world_size()
    e, topk, d, i, act, bias = MODELS[model]
    config = get_2stage_cfgs(tokens, world, e, topk, d, i, activation=act, bias_dtype=bias)
    torch.manual_seed(args.seed + rank)
    local = config.local_experts
    w13 = torch.randint(0, 256, (local, 2 * i, d // 2), dtype=torch.uint8, device="cuda")
    w2 = torch.randint(0, 256, (local, d, i // 2), dtype=torch.uint8, device="cuda")
    s13 = torch.full((local, 2 * i, d // 32), 119, dtype=torch.uint8, device="cuda")
    s2 = torch.full((local, d, i // 32), 119, dtype=torch.uint8, device="cuda")
    b1 = torch.zeros((local, 2, i), dtype=torch.bfloat16, device="cuda") if bias == "bf16" else None
    b2 = torch.zeros((local, d), dtype=torch.bfloat16, device="cuda") if bias == "bf16" else None
    weights = pack_expert_weights(config, w13, w2, s13, s2, b1, b2)
    del w13, w2, s13, s2
    x = torch.randn((tokens, d), dtype=torch.bfloat16, device="cuda")
    workspace = MegaMoeWorkspace.allocate(config)
    views = workspace.input_views(tokens)
    views.expert_ids.copy_(torch.rand((tokens, e), device="cuda").topk(topk, dim=-1).indices.int())
    views.expert_weights.copy_(torch.rand((tokens, topk), device="cuda").softmax(-1))
    workspace.quantize(x)

    def compute():
        return workspace.run(weights, tokens)

    def dynamic():
        workspace.quantize(x)
        return compute()

    assert torch.isfinite(compute()).all()
    functions = {"compute": compute, "dynamic": dynamic}
    timings = {}
    for scope in args.scopes:
        timings[scope] = measure(functions[scope], args)
        if rank == 0:
            print(f"{model} EP{world} M{tokens} {scope}: {timings[scope]['median_us']:.2f} us", file=sys.stderr)
    dist.barrier()
    return {"model": model, "tokens_per_rank": tokens, "solution_id": int(config.solution), "timings": timings}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=["gptoss"])
    parser.add_argument("--tokens", nargs="+", type=int, default=[8, 128, 1024])
    parser.add_argument("--scopes", nargs="+", choices=["compute", "dynamic"], default=["dynamic"])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--graph-iters", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, help="Optional JSON output; defaults to rank-zero stdout")
    args = parser.parse_args()
    if args.warmup < 1 or args.repeat < 1 or args.graph_iters < 1 or any(not 0 <= m <= 1024 for m in args.tokens):
        parser.error("warmup/repeat/graph-iters must be positive; tokens must be in [0, 1024]")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("gloo")
    try:
        report = {
            "backend": "avelang",
            "world_size": dist.get_world_size(),
            "input": "synthetic; fixed top-k routing",
            "gpu": torch.cuda.get_device_name(),
            "seed": args.seed,
            "timing": {
                "warmup": args.warmup,
                "repeat": args.repeat,
                "graph_iters": args.graph_iters,
                "statistic": "median of per-sample maximum rank latency",
            },
            "cases": [run_case(args, model, m) for model in args.models for m in args.tokens],
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
