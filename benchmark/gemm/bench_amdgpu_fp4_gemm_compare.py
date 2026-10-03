#!/usr/bin/env python3
"""Optional Petit integration benchmark: tune FP4 GEMM on shared inputs."""

import argparse
import hashlib
import importlib.util
import json
import math
import random
import re
import statistics
import subprocess
import sys
from pathlib import Path

import torch

from avelang_kernels.amdgpu import fp4_gemm_transposed_b
from avelang_kernels.amdgpu.fp4_gemm.solution import (
    MatmulElementB, MatmulFeatures, MatmulMfmaType, SolutionId, available_solutions,
)
from avelang_kernels.amdgpu.fp4_gemm.utils import dequantize_fp4, process_fp4_scales, repack_fp4


def _load_petit(path):
    spec = importlib.util.spec_from_file_location("petit_kernel.ops", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load Petit from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _common_solutions(n, k, dtype, element_b, high_precision, solution_list):
    mfma = MatmulMfmaType.FP16 if dtype == torch.float16 else MatmulMfmaType.BF16
    features = MatmulFeatures.GRID
    if high_precision:
        features |= MatmulFeatures.HIGH_PRECISION
    ave = {int(s) for s in available_solutions(
        n, k, element_b=element_b, mfma_type=mfma, high_precision=high_precision,
    )}
    petit = set()
    for value in re.findall(r"PETIT_KERNEL_IMPL\(([0-9a-f]+)\)", solution_list.read_text()):
        solution = SolutionId.from_int(int(value, 16))
        if (
            solution.element_b == element_b and solution.mfma_type == mfma
            and solution.features == features and int(solution.warp_partition) == 0
            and n % solution.group_n == 0 and k % solution.group_k == 0
        ):
            petit.add(int(solution))
    if ave != petit:
        raise ValueError(f"Candidate sets differ: AveLang-only={sorted(ave - petit)}, Petit-only={sorted(petit - ave)}")
    if not ave:
        raise ValueError(f"No solutions for N={n}, K={k}")
    return sorted(ave)


def _fingerprint(tensor):
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).flatten().numpy()
    return hashlib.sha256(memoryview(raw)).hexdigest()


def _prepare_graph(fn, reference, warmup, tolerance):
    for _ in range(max(1, warmup)):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.graph(graph, stream=stream):
        output = fn()
    torch.cuda.synchronize()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output, reference, rtol=tolerance, atol=tolerance)
    return graph, output


def _elapsed(graph, iters):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iters


def _measure_pair(graphs, args, first):
    probe_ms = max(_elapsed(graph, 3) for graph in graphs.values())
    iters = args.iters or min(args.max_iters, max(1, math.ceil(args.sample_ms / probe_ms)))
    samples = {name: [] for name in graphs}
    order = (first, "petit" if first == "avelang" else "avelang")
    for repeat in range(args.repeat):
        for name in order if repeat % 2 == 0 else order[::-1]:
            samples[name].append(_elapsed(graphs[name], iters))
    return iters, samples


def _make_problem(m, n, k, seed, dtype, element_b, petit):
    # Generate once: both kernels receive this very same A and global_scale.
    torch.manual_seed(seed)
    a = torch.randn((m, k), dtype=dtype, device="cuda")
    qweight = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device="cuda")
    if element_b == MatmulElementB.MXFP4:
        scales = torch.randint(124, 131, (n, k // 32), dtype=torch.uint8, device="cuda")
        process_scales = petit.process_mxfp4_scales
    else:
        scales = (torch.rand((n, k // 16), device="cuda") * 3.5 + 0.25).to(torch.float8_e4m3fn)
        process_scales = petit.process_nvfp4_scales
    global_scale = torch.tensor([0.25], dtype=torch.float32, device="cuda")
    hashes = {name: _fingerprint(value) for name, value in (
        ("a", a), ("qweight", qweight), ("scales", scales), ("global_scale", global_scale),
    )}
    ave_weight, ave_scales = repack_fp4(qweight), process_fp4_scales(scales, element_b=element_b)
    petit_weight = petit.repack_nvfp4(qweight.view(torch.int32), n, k)
    petit_scales = process_scales(scales, n, k)
    # Tensor shapes/dtypes differ, but their packed storage must be byte-identical.
    for name, lhs, rhs in (("weight", ave_weight, petit_weight), ("scale", ave_scales, petit_scales)):
        if not torch.equal(lhs.view(torch.uint8).flatten(), rhs.view(torch.uint8).flatten()):
            raise AssertionError(f"{name} packing differs between AveLang and Petit")
    for name, value in (("qweight", qweight), ("scales", scales)):
        if _fingerprint(value) != hashes[name]:
            raise AssertionError(f"Packing mutated the shared {name}")
    weight = dequantize_fp4(qweight, scales, element_b=element_b)
    weight.mul_(global_scale)
    reference = (a.float() @ weight.T).to(dtype)
    return a, global_scale, ave_weight, ave_scales, petit_weight, petit_scales, reference, hashes


def _tune_case(args, petit, m, n, k, seed):
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    element_b = MatmulElementB.MXFP4 if args.format == "mxfp4" else MatmulElementB.NVFP4
    mul = petit.mul_mxfp4_a16 if element_b == MatmulElementB.MXFP4 else petit.mul_nvfp4_a16
    solutions = _common_solutions(n, k, dtype, element_b, args.high_precision, args.solution_list)
    if args.solution_id:
        missing = set(args.solution_id) - set(solutions)
        if missing:
            raise ValueError(f"Unavailable requested solutions: {sorted(missing)}")
        solutions = sorted(set(args.solution_id))
    random.Random(seed).shuffle(solutions)
    a, scale, aw, ass, pw, ps, reference, hashes = _make_problem(m, n, k, seed, dtype, element_b, petit)
    out = torch.empty_like(reference)
    tolerance = 2e-3 if dtype == torch.float16 else 2e-2
    results = {"avelang": [], "petit": []}
    matches = []
    print(f"{m}x{n}x{k}: shared inputs and packed bytes verified; {len(solutions)} solutions", flush=True)
    for index, value in enumerate(solutions):
        solution = SolutionId.from_int(value)

        def run_ave():
            return fp4_gemm_transposed_b(a, aw, ass, scale, out=out, solution_id=solution, element_b=element_b)

        def run_petit():
            return mul(a, pw, ps, scale, m, n, k, value)

        functions = {"avelang": run_ave, "petit": run_petit}
        graphs, outputs = {}, {}
        first = "avelang" if index % 2 == 0 else "petit"
        order = (first, "petit" if first == "avelang" else "avelang")
        for name in order:
            graphs[name], outputs[name] = _prepare_graph(functions[name], reference, args.warmup, tolerance)
        torch.testing.assert_close(outputs["avelang"], outputs["petit"], rtol=tolerance, atol=tolerance)
        bitwise_equal = torch.equal(
            outputs["avelang"].view(torch.uint8), outputs["petit"].view(torch.uint8),
        )
        iters, samples = _measure_pair(graphs, args, first)
        times = {name: statistics.median(values) for name, values in samples.items()}
        for name in results:
            results[name].append({
                "solution_id": f"0x{value:012x}", "iters": iters,
                "time_ms": times[name], "samples_ms": samples[name],
                "tflops": 2 * m * n * k / (times[name] * 1e9),
                "tile_shape": [solution.tile_m, solution.tile_n, solution.tile_k * 4],
            })
        matches.append({
            "solution_id": f"0x{value:012x}", "outputs_bitwise_equal": bitwise_equal,
            "avelang_over_petit": times["avelang"] / times["petit"],
        })
        print(
            f"  [{index + 1:02d}/{len(solutions)}] 0x{value:012x}: "
            f"AveLang {times['avelang']:.4f} ms, Petit {times['petit']:.4f} ms; "
            f"outputs bitwise equal={bitwise_equal}", flush=True,
        )
        del graphs, outputs
    for name, value in (("a", a), ("global_scale", scale)):
        if _fingerprint(value) != hashes[name]:
            raise AssertionError(f"GEMM mutated the shared {name}")
    for name, values in results.items():
        values.sort(key=lambda result: result["time_ms"])
        print(f"  {name} top {args.top_k}:", flush=True)
        for rank, result in enumerate(values[:args.top_k], 1):
            print(f"    {rank}: {result['solution_id']} {result['time_ms']:.4f} ms {result['tflops']:.2f} TFLOPS")
    return {
        "m": m, "n": n, "k": k, "seed": seed, "input_sha256": hashes,
        "packed_inputs_byte_identical": True, "results": results, "same_solution": matches,
        "best_avelang_over_petit": results["avelang"][0]["time_ms"] / results["petit"][0]["time_ms"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--petit-root", type=Path, default=Path.home() / "petit-kernel")
    parser.add_argument("--sizes", nargs="+", type=int, default=[1024, 2048, 4096, 8192, 16384])
    parser.add_argument("--m", type=int)
    parser.add_argument("--n", type=int)
    parser.add_argument("--k", type=int)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--format", choices=("nvfp4", "mxfp4"), default="nvfp4")
    parser.add_argument("--high-precision", action="store_true")
    parser.add_argument("--solution-id", type=lambda value: int(value, 0), nargs="+")
    parser.add_argument("--device", type=int, default=0, help="Logical device after HIP_VISIBLE_DEVICES filtering")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--sample-ms", type=float, default=50.0)
    parser.add_argument("--max-iters", type=int, default=500)
    parser.add_argument("--iters", type=int)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Combined JSON with both top-k lists and input hashes",
    )
    args = parser.parse_args()
    if args.format == "mxfp4" and args.dtype == "float16":
        parser.error("MXFP4 requires --dtype bfloat16")
    if not torch.cuda.is_available() or torch.version.hip is None:
        parser.error("Requires PyTorch ROCm")
    shape = (args.m, args.n, args.k)
    if any(value is not None for value in shape) and not all(value is not None for value in shape):
        parser.error("Specify --m, --n and --k together")
    shapes = [shape] if shape[0] is not None else [(size, size, size) for size in args.sizes]
    if any(min(shape) <= 0 or shape[1] % 64 or shape[2] % 256 for shape in shapes):
        parser.error("Positive dimensions required; N divisible by 64 and K divisible by 256")
    if args.warmup < 1 or min(args.repeat, args.max_iters, args.top_k, args.sample_ms) <= 0:
        parser.error("warmup/repeat/max-iters/top-k/sample-ms must be positive")
    if args.iters is not None and args.iters <= 0:
        parser.error("iters must be positive")
    usage = subprocess.run(["rocm-smi", "--showuse", "--showmemuse"], capture_output=True, text=True, check=True)
    print(usage.stdout, file=sys.stderr, flush=True)
    torch.cuda.set_device(args.device)
    torch.set_float32_matmul_precision("highest")
    props = torch.cuda.get_device_properties(args.device)
    args.petit_root = args.petit_root.expanduser().resolve()
    library = args.petit_root / "build/lib/pybind/petit_kernels.so"
    args.solution_list = args.petit_root / "build/lib/gemm/rocm/quantization/fp4/solutions.inl"
    petit = _load_petit(library)
    payload = {
        "schema_version": 1, "operation": f"{args.format}_gemm", "backend": "avelang_vs_petit",
        "device": {"name": props.name, "arch": props.gcnArchName, "logical_index": args.device},
        "dtype": args.dtype, "format": args.format, "high_precision": args.high_precision,
        "petit_library": {"path": str(library), "sha256": hashlib.sha256(library.read_bytes()).hexdigest()},
        "timing": {name: getattr(args, name) for name in ("warmup", "repeat", "sample_ms", "max_iters", "iters")},
        "cases": [],
    }
    args.output = args.output.expanduser()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for index, shape in enumerate(shapes):
        payload["cases"].append(_tune_case(args, petit, *shape, args.seed + index))
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print("\nBest tuned results (AveLang/Petit time; >1 means AveLang slower):")
    for case in payload["cases"]:
        print(f"  {case['m']}x{case['n']}x{case['k']}: {case['best_avelang_over_petit']:.3f}x")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
