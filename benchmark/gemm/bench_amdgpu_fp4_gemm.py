#!/usr/bin/env python3
import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import torch
from avelang_kernels.amdgpu import fp4_gemm
from avelang_kernels.amdgpu.fp4_gemm_utils import (
    dequantize_fp4,
    process_fp4_scales,
    repack_fp4,
)
from avelang_kernels.amdgpu.fp4_gemm_solution import (
    MatmulElementB,
    MatmulMfmaType,
    available_solutions,
)


def _capture(fn):
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    with torch.cuda.graph(graph, stream=stream):
        fn()
    torch.cuda.synchronize()
    return graph


def _elapsed(graph, iters: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iters


def _make_problem(m: int, n: int, k: int, seed: int):
    torch.manual_seed(seed)
    a = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    qweight = torch.randint(
        0, 256, (n, k // 2), dtype=torch.uint8, device="cuda"
    )
    scales = (torch.rand((n, k // 16), device="cuda") * 3.5 + 0.25).to(
        torch.float8_e4m3fn
    )
    packed_weight = repack_fp4(qweight)
    packed_scales = process_fp4_scales(scales)
    global_scale = 0.25

    weight = dequantize_fp4(qweight, scales)
    weight.mul_(global_scale)
    reference = (a.float() @ weight.T).to(torch.bfloat16)
    del weight

    out = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
    return a, packed_weight, packed_scales, global_scale, out, reference


def _benchmark_solution(
    args,
    solution,
    a,
    packed_weight,
    packed_scales,
    global_scale,
    out,
    reference,
):
    def run_kernel():
        return fp4_gemm(
            a,
            packed_weight,
            packed_scales,
            global_scale,
            out=out,
            solution_id=solution,
        )

    for _ in range(args.warmup):
        run_kernel()
    torch.cuda.synchronize()

    graph = _capture(run_kernel)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out, reference, rtol=2e-2, atol=2e-2)

    probe_ms = _elapsed(graph, 1)
    iters = args.iters or min(
        args.max_iters, max(1, math.ceil(args.sample_ms / probe_ms))
    )
    samples = [_elapsed(graph, iters) for _ in range(args.repeat)]
    time_ms = statistics.median(samples)
    m, k = a.shape
    n = out.shape[1]
    tflops = 2.0 * m * n * k / (time_ms * 1.0e9)
    shape = solution.shape
    return {
        "solution_id": f"0x{int(solution):012x}",
        "tile_shape": [shape.tile_m, shape.tile_n, shape.tile_k],
        "warp_partition": [
            shape.warp_partition_m,
            shape.warp_partition_n,
            shape.warp_partition_k,
        ],
        "iters": iters,
        "time_ms": time_ms,
        "tflops": tflops,
    }


def _tune_case(args, m: int, n: int, k: int, seed: int):
    problem = _make_problem(m, n, k, seed)
    solutions = available_solutions(
        n,
        k,
        element_b=MatmulElementB.NVFP4,
        mfma_type=MatmulMfmaType.BF16,
    )
    results = []
    for index, solution in enumerate(solutions, 1):
        result = _benchmark_solution(args, solution, *problem)
        results.append(result)
        print(
            f"tune {m}x{n}x{k} [{index:02d}/{len(solutions)}] "
            f"{result['solution_id']}: {result['time_ms']:.4f} ms, "
            f"{result['tflops']:.2f} TFLOPS",
            file=sys.stderr,
            flush=True,
        )
    results.sort(key=lambda result: result["time_ms"])
    return {"m": m, "n": n, "k": k, "seed": seed, "results": results}


def _print_top(cases, top_k: int):
    for case in cases:
        print(
            f"\nAveLang top {top_k}: M={case['m']}, N={case['n']}, K={case['k']}",
            file=sys.stderr,
        )
        for rank, result in enumerate(case["results"][:top_k], 1):
            print(
                f"  {rank}: {result['solution_id']}  "
                f"{result['time_ms']:.4f} ms  {result['tflops']:.2f} TFLOPS",
                file=sys.stderr,
            )


def run(args):
    if not args.tune:
        raise ValueError("This benchmark only supports tuning; pass --tune.")
    if not torch.cuda.is_available() or torch.version.hip is None:
        raise RuntimeError("This benchmark requires PyTorch ROCm.")
    if args.repeat < 1 or args.warmup < 0 or args.max_iters < 1:
        raise ValueError("repeat/max-iters must be positive and warmup non-negative.")
    if args.iters is not None and args.iters < 1:
        raise ValueError("iters must be positive.")
    if args.top_k < 1:
        raise ValueError("top-k must be positive.")

    single_shape = (args.m, args.n, args.k)
    if any(value is not None for value in single_shape):
        if not all(value is not None for value in single_shape):
            raise ValueError("--m, --n and --k must be specified together.")
        shapes = [single_shape]
    else:
        shapes = [(size, size, size) for size in args.sizes]
    if any(min(shape) <= 0 for shape in shapes):
        raise ValueError("All dimensions must be positive.")

    cases = [
        _tune_case(args, *shape, args.seed + index)
        for index, shape in enumerate(shapes)
    ]
    _print_top(cases, args.top_k)
    payload = {
        "schema_version": 1,
        "operation": "fp4_gemm",
        "backend": "avelang",
        "device": torch.cuda.get_device_name(torch.cuda.current_device()),
        "dtypes": {
            "a": "bfloat16",
            "weight": "fp4_e2m1",
            "scale": "float8_e4m3fn",
            "output": "bfloat16",
        },
        "timing": {
            "warmup": args.warmup,
            "repeat": args.repeat,
            "sample_ms": args.sample_ms,
            "max_iters": args.max_iters,
            "fixed_iters": args.iters,
            "statistic": "median",
        },
        "cases": cases,
    }
    serialized = json.dumps(payload, indent=2) + "\n"
    if args.output is None:
        print(serialized, end="")
    else:
        output = args.output.expanduser()
        output.write_text(serialized, encoding="utf-8")
        print(f"wrote {output}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Tune AveLang AMDGPU FP4 GEMM")
    parser.add_argument("--tune", action="store_true")
    parser.add_argument(
        "--sizes", nargs="+", type=int, default=[1024, 2048, 4096, 8192, 16384]
    )
    parser.add_argument("--m", type=int)
    parser.add_argument("--n", type=int)
    parser.add_argument("--k", type=int)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--iters", type=int, help="Fixed graph replays per sample")
    parser.add_argument("--sample-ms", type=float, default=50.0)
    parser.add_argument("--max-iters", type=int, default=500)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, help="JSON output path; defaults to stdout")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
