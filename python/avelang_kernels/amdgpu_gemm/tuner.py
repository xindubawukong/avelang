from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import GemmConfig
from .kernel import gemm_pipeline_transposed_b


@dataclass(frozen=True)
class BenchmarkResult:
    config: GemmConfig
    elapsed_ms: float
    tflops: float
    bandwidth_gbs: float


def benchmark_config(
    config: GemmConfig,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    *,
    warmup: int,
    repeat: int,
    iters: int,
) -> BenchmarkResult:
    m = A.shape[0]
    k = A.shape[1]
    n = B.shape[0]

    for _ in range(max(1, warmup)):
        gemm_pipeline_transposed_b(A, B, out=C, config=config)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    capture_stream = torch.cuda.Stream()
    with torch.cuda.graph(graph, stream=capture_stream):
        gemm_pipeline_transposed_b(A, B, out=C, config=config)
    torch.cuda.synchronize()

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    start_evt.record()
    for _ in range(repeat):
        for _ in range(iters):
            graph.replay()
    end_evt.record()
    torch.cuda.synchronize()

    elapsed_ms_total = start_evt.elapsed_time(end_evt)
    elapsed_s = (elapsed_ms_total * 1.0e-3) / (repeat * iters)

    flops = 2.0 * m * n * k
    tflops = flops / elapsed_s / 1.0e12
    bytes_moved = (m * k + k * n + m * n) * 2
    bandwidth_gbs = bytes_moved / elapsed_s / 1.0e9

    return BenchmarkResult(
        config=config,
        elapsed_ms=elapsed_s * 1.0e3,
        tflops=tflops,
        bandwidth_gbs=bandwidth_gbs,
    )
