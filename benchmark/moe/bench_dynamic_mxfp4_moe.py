"""Benchmark AveLang local MoE with synthetic inputs and fixed routing."""

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch
from avelang_kernels.amdgpu.local_moe import ExpertWeights, MoeWorkspace, Routing, dynamic_mxfp4_moe, get_2stage_cfgs
from avelang_kernels.amdgpu.local_moe.api import prepare_input
from avelang_kernels.amdgpu.local_moe.route_reduce import make_route_reduce
from avelang_kernels.amdgpu.local_moe.stage1 import make_stage1
from avelang_kernels.amdgpu.local_moe.stage2 import make_stage2

BENCH_CASES = {
    "gptoss": {
        "model_dim": 3072,
        "inter_dim": 3072,
        "expert": 16,
        "topk": 4,
        "activation": "swiglu",
        "bias_dtype": "bf16",
    },
    "dsv3": {"model_dim": 7168, "inter_dim": 2048, "expert": 33, "topk": 9, "activation": "silu", "bias_dtype": "none"},
    "dsv4": {"model_dim": 7168, "inter_dim": 3072, "expert": 49, "topk": 7, "activation": "silu", "bias_dtype": "none"},
    "kimi_k3": {
        "model_dim": 3584,
        "inter_dim": 384,
        "expert": 896,
        "topk": 16,
        "activation": "situ",
        "bias_dtype": "none",
    },
}


def make_inputs(config, tokens, seed):
    torch.manual_seed(seed)
    e, d, i, k = (config.experts, config.hidden, config.intermediate, config.topk)
    x = torch.randn((tokens, d), dtype=torch.bfloat16, device="cuda")
    w13 = torch.randint(0, 256, (e, 2 * i, d // 2), dtype=torch.uint8, device="cuda")
    w2 = torch.randint(0, 256, (e, d, i // 2), dtype=torch.uint8, device="cuda")
    s13 = torch.randint(117, 123, (e, 2 * i, d // 32), dtype=torch.uint8, device="cuda")
    s2 = torch.randint(117, 123, (e, d, i // 32), dtype=torch.uint8, device="cuda")
    b1 = torch.randn((e, 2, i), dtype=torch.bfloat16, device="cuda") if config.bias else None
    b2 = torch.randn((e, d), dtype=torch.bfloat16, device="cuda") if config.bias else None
    weights = ExpertWeights.pack(w13, w2, s13, s2, b1, b2)
    top_ids = torch.rand((tokens, e)).topk(k, dim=1).indices
    top_weights = torch.rand((tokens, k)).softmax(-1)
    bm = config.stage1_tile_m
    capacity = (tokens * k + e * bm - k + bm - 1) // bm * bm
    ids = torch.full((capacity,), k << 24 | tokens, dtype=torch.int32)
    rw = torch.zeros(capacity, dtype=torch.float32)
    experts = torch.full((capacity // bm,), -1, dtype=torch.int32)
    begin = 0
    for expert in range(e):
        matches = (top_ids == expert).nonzero()
        count = len(matches)
        ids[begin : begin + count] = matches[:, 0].int() | matches[:, 1].int() << 24
        rw[begin : begin + count] = top_weights[matches[:, 0], matches[:, 1]]
        blocks = (count + bm - 1) // bm
        experts[begin // bm : begin // bm + blocks] = expert
        begin += blocks * bm
    routing = Routing(
        ids.cuda(), rw.cuda(), experts.cuda(), torch.tensor([begin, tokens], dtype=torch.int32, device="cuda")
    )
    return (x, weights, routing)


def measure(fn, warmup, repeat, graph_iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(graph_iters):
            fn()
    samples = []
    for _ in range(repeat):
        start, end = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / graph_iters)
    return {"median_us": statistics.median(samples), "min_us": min(samples), "max_us": max(samples)}


def run_case(args, model, tokens):
    config = get_2stage_cfgs(token=tokens, **BENCH_CASES[model], arch="gfx950")
    x, weights, routing = make_inputs(config, tokens, args.seed)
    ws = MoeWorkspace.allocate(x, routing, config)
    d = config.hidden
    stage1_kernel, stage2_kernel = (make_stage1(config), make_stage2(config))

    def stage1():
        stage1_kernel[lambda: config.stage1_grid(routing.capacity)](
            ws.input_act,
            ws.sorted_scales,
            weights.w13,
            weights.s13,
            weights.bias1 if config.bias else ws.out,
            routing.ids,
            routing.experts,
            routing.counts,
            ws.intermediate,
            routing.capacity,
            num_warps=config.stage1_num_warps,
        )

    def stage2():
        if not config.use_route_reduce:
            ws.out.zero_()
        stage2_kernel[lambda: config.stage2_grid(tokens, routing.capacity)](
            ws.intermediate,
            weights.w2,
            weights.s2,
            weights.bias2 if config.bias else ws.out,
            routing.ids,
            routing.experts,
            routing.weights,
            routing.counts,
            ws.route_output if config.use_route_reduce else ws.out,
            routing.capacity,
            num_warps=config.stage2_num_warps,
        )
        if config.use_route_reduce:
            make_route_reduce(d, config.topk)[lambda: ((tokens, (d // 8 + 511) // 512, 1), (512, 1, 1))](
                ws.route_output, ws.out, tokens, num_warps=8
            )
        return ws.out

    def compute():
        stage1()
        return stage2()

    def dynamic():
        return dynamic_mxfp4_moe(x, weights, routing, config, workspace=ws)

    prepare_input(x, routing, config, ws)
    assert torch.isfinite(compute()).all()
    functions = {"stage1": stage1, "stage2": stage2, "compute": compute, "dynamic": dynamic}
    timings = {}
    for scope in args.scopes:
        timings[scope] = measure(functions[scope], args.warmup, args.repeat, args.graph_iters)
        print(f"{model} M{tokens} {scope}: {timings[scope]['median_us']:.2f} us", file=sys.stderr)
    return {"model": model, "tokens": tokens, "solution_id": int(config.solution), "timings": timings}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=BENCH_CASES, default=["gptoss"])
    parser.add_argument("--tokens", nargs="+", type=int, default=[8, 128, 1024])
    parser.add_argument("--scopes", nargs="+", choices=["stage1", "stage2", "compute", "dynamic"], default=["dynamic"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--graph-iters", type=int, default=16)
    parser.add_argument("--output", type=Path, help="Optional JSON output; defaults to stdout")
    args = parser.parse_args()
    if args.warmup < 1 or args.repeat < 1 or args.graph_iters < 1 or any(m <= 0 for m in args.tokens):
        parser.error("tokens, warmup, repeat and graph-iters must be positive")
    report = {
        "backend": "avelang",
        "input": "synthetic; fixed expert-sorted routing",
        "gpu": torch.cuda.get_device_name(),
        "seed": args.seed,
        "timing": {"warmup": args.warmup, "repeat": args.repeat, "graph_iters": args.graph_iters},
        "cases": [run_case(args, model, m) for model in args.models for m in args.tokens],
    }
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output:
        path = args.output.expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encoded)
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
