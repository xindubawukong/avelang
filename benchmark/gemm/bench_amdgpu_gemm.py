#!/usr/bin/env python3
import argparse

import torch

from avelang_kernels import amdgpu_gemm


def _ensure_rocm_available(label):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    if not hasattr(torch.version, "hip") or torch.version.hip is None:
        raise RuntimeError(f"HIP is not available; {label} benchmark requires ROCm.")


def _select_configs(m, n, k, algo, config_name):
    if config_name is not None:
        config = amdgpu_gemm.get_config(config_name)
        if not config.supports(m, n, k):
            raise ValueError(f"Config {config.name!r} does not support M={m}, N={n}, K={k}.")
        return [config]

    if algo == "tune":
        configs = amdgpu_gemm.enumerate_configs(m, n, k)
        if not configs:
            raise ValueError(f"No AMDGPU GEMM config supports M={m}, N={n}, K={k}.")
        return configs

    return [amdgpu_gemm.default_config(m, n, k)]


def _validate_result(A, B, C):
    expected = (A.float() @ B.float().T).to(dtype=torch.bfloat16, device="cpu")
    actual = C.to("cpu")
    max_abs = torch.max(torch.abs(actual - expected)).item()
    if not torch.allclose(actual, expected, rtol=1e-1, atol=1e-1):
        raise AssertionError(f"Validation failed (max abs diff {max_abs}).")
    print(f"validation=max_abs_diff:{max_abs:.6f}")


def _print_result(m, n, k, result):
    config = result.config
    print(
        f"M={m} N={n} K={k} config={config.name} "
        f"time_ms={result.elapsed_ms:.4f} tflops={result.tflops:.3f} "
        f"bandwidth_gbs={result.bandwidth_gbs:.3f}"
    )


def run_gemm_pipeline_benchmark(
    m,
    n,
    k,
    warmup,
    repeat,
    iters,
    validate,
    algo,
    config_name,
):
    _ensure_rocm_available("gemm_pipeline_transposed_b")
    configs = _select_configs(m, n, k, algo, config_name)

    A = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    B = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")
    C = torch.zeros((m, n), dtype=torch.bfloat16, device="cuda")

    results = []
    for config in configs:
        try:
            result = amdgpu_gemm.benchmark_config(
                config,
                A,
                B,
                C,
                warmup=warmup,
                repeat=repeat,
                iters=iters,
            )
        except Exception as exc:
            if algo != "tune":
                raise
            print(f"config={config.name} failed={type(exc).__name__}: {exc}")
            continue
        results.append(result)

    if not results:
        raise RuntimeError(f"No AMDGPU GEMM config completed for M={m}, N={n}, K={k}.")

    if len(results) == 1:
        best = results[0]
        _print_result(m, n, k, best)
    else:
        results.sort(key=lambda result: result.tflops, reverse=True)
        for result in results:
            _print_result(m, n, k, result)
        best = results[0]
        print(f"best_config={best.config.name} tflops={best.tflops:.3f}")

    if validate:
        amdgpu_gemm.gemm_pipeline_transposed_b(A, B, out=C, config=best.config)
        torch.cuda.synchronize()
        _validate_result(A, B, C)


def main():
    parser = argparse.ArgumentParser(
        description="AMDGPU GEMM pipeline benchmark"
    )
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--algo", choices=("default", "tune"), default="default")
    parser.add_argument("--config", choices=sorted(amdgpu_gemm.CONFIG_BY_NAME), default=None)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--iters", type=int, default=100, help="Graph replays per timing repeat")
    parser.add_argument("--validate", action="store_true", default=False)
    parser.add_argument("--list-configs", action="store_true", default=False)
    args = parser.parse_args()

    if args.list_configs:
        for config in amdgpu_gemm.CONFIGS:
            print(f"{config.name}: {config.source}")
        return

    run_gemm_pipeline_benchmark(
        args.m,
        args.n,
        args.k,
        args.warmup,
        args.repeat,
        args.iters,
        args.validate,
        args.algo,
        args.config,
    )


if __name__ == "__main__":
    main()
