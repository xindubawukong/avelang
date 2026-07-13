#!/usr/bin/env python3
import argparse

import torch

from avelang_kernels import amdgpu_gemm


def _ensure_rocm_available(label):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    if not hasattr(torch.version, "hip") or torch.version.hip is None:
        raise RuntimeError(f"HIP is not available; {label} benchmark requires ROCm.")


def _validate_result(A, B, C):
    expected = (A.float() @ B.float().T).to(dtype=torch.bfloat16, device="cpu")
    actual = C.to("cpu")
    max_abs = torch.max(torch.abs(actual - expected)).item()
    if not torch.allclose(actual, expected, rtol=1e-1, atol=1e-1):
        raise AssertionError(f"Validation failed (max abs diff {max_abs}).")
    print(f"validation=max_abs_diff:{max_abs:.6f}")


def _benchmark(A, B, C, config, warmup, repeat, iters):
    for _ in range(max(1, warmup)):
        amdgpu_gemm.gemm_pipeline_transposed_b(A, B, out=C, config=config)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    capture_stream = torch.cuda.Stream()
    with torch.cuda.graph(graph, stream=capture_stream):
        amdgpu_gemm.gemm_pipeline_transposed_b(A, B, out=C, config=config)
    torch.cuda.synchronize()

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    start_evt.record()
    for _ in range(repeat):
        for _ in range(iters):
            graph.replay()
    end_evt.record()
    torch.cuda.synchronize()

    elapsed_ms = start_evt.elapsed_time(end_evt) / (repeat * iters)
    return elapsed_ms


def _print_result(m, n, k, config, elapsed_ms):
    elapsed_s = elapsed_ms * 1.0e-3
    tflops = 2.0 * m * n * k / elapsed_s / 1.0e12
    bytes_moved = (m * k + k * n + m * n) * 2
    bandwidth_gbs = bytes_moved / elapsed_s / 1.0e9
    print(
        f"M={m} N={n} K={k} config={config.name} "
        f"time_ms={elapsed_ms:.4f} tflops={tflops:.3f} "
        f"bandwidth_gbs={bandwidth_gbs:.3f}"
    )


def run_gemm_pipeline_benchmark(
    m,
    n,
    k,
    warmup,
    repeat,
    iters,
    validate,
):
    _ensure_rocm_available("gemm_pipeline_transposed_b")
    config = amdgpu_gemm.default_config(m, n, k)

    A = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    B = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")
    C = torch.zeros((m, n), dtype=torch.bfloat16, device="cuda")

    elapsed_ms = _benchmark(A, B, C, config, warmup, repeat, iters)
    _print_result(m, n, k, config, elapsed_ms)

    if validate:
        amdgpu_gemm.gemm_pipeline_transposed_b(A, B, out=C, config=config)
        torch.cuda.synchronize()
        _validate_result(A, B, C)


def main():
    parser = argparse.ArgumentParser(
        description="AMDGPU GEMM pipeline benchmark"
    )
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--iters", type=int, default=100, help="Graph replays per timing repeat")
    parser.add_argument("--validate", action="store_true", default=False)
    args = parser.parse_args()

    run_gemm_pipeline_benchmark(
        args.m,
        args.n,
        args.k,
        args.warmup,
        args.repeat,
        args.iters,
        args.validate,
    )


if __name__ == "__main__":
    main()
