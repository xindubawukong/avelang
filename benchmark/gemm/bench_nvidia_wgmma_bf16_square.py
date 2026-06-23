#!/usr/bin/env python3
import argparse

import torch

import avelang
import avelang.language as S


WGMMA_SWIZZLE_128B = S.WGMMA_SWIZZLE_128B

TILE_M = 64
TILE_N = 64
TILE_K = 64
THREADS_PER_CTA = 128
BF16_BYTES = 2
CP_ASYNC_BYTES = 16
BF16_PER_CP_ASYNC = CP_ASYNC_BYTES // BF16_BYTES
CP_ASYNC_CHUNKS_PER_ROW = TILE_K // BF16_PER_CP_ASYNC
CP_ASYNC_CHUNKS = (TILE_M * TILE_K) // BF16_PER_CP_ASYNC
STORE_ITERS = (TILE_M * TILE_N) // THREADS_PER_CTA

DEFAULT_SIZES = (1024, 2048, 4096, 8192, 16384)
DEFAULT_WARMUP = 3


@avelang.jit
def gemm_wgmma_bf16_square_kernel(
    a_ptr: S.Pointer(S.bf16),
    b_ptr: S.Pointer(S.bf16),
    c_ptr: S.Pointer(S.bf16),
    size: S.constexpr,
):
    tid = S.thread_id(0)
    block_m = S.block_id(1) * TILE_M
    block_n = S.block_id(0) * TILE_N

    a = S.make_tensor(a_ptr, S.bf16, S.make_layout((size, size), (size, 1)))
    b = S.make_tensor(b_ptr, S.bf16, S.make_layout((size, size), (size, 1)))
    c = S.make_tensor(c_ptr, S.bf16, S.make_layout((size, size), (size, 1)))

    a_shared = S.make_shared((TILE_M, TILE_K), S.bf16)
    b_shared = S.make_shared((TILE_K, TILE_N), S.bf16)
    c_shared = S.make_shared((TILE_M, TILE_N), S.f32)

    k_tiles = size // TILE_K

    for k_tile in S.range(k_tiles):
        k_base = k_tile * TILE_K

        for i in S.range(CP_ASYNC_CHUNKS // THREADS_PER_CTA):
            chunk = tid + i * THREADS_PER_CTA
            row = chunk // CP_ASYNC_CHUNKS_PER_ROW
            col = (chunk % CP_ASYNC_CHUNKS_PER_ROW) * BF16_PER_CP_ASYNC
            swizzled_col = col ^ ((row % 8) * 8)
            dst_offset = (row * TILE_K + swizzled_col) * BF16_BYTES

            a_src_offset = ((block_m + row) * size + k_base + col) * BF16_BYTES
            b_src_offset = ((k_base + row) * size + block_n + col) * BF16_BYTES

            S.nvvm.cp_async_ca_shared_global(
                a_shared, a, dst_offset, a_src_offset, CP_ASYNC_BYTES
            )
            S.nvvm.cp_async_ca_shared_global(
                b_shared, b, dst_offset, b_src_offset, CP_ASYNC_BYTES
            )

        S.nvvm.cp_async_commit_group()
        S.nvvm.cp_async_wait_group(0)
        S.syncthreads()

        desc_a = S.nvvm.make_wgmma_descriptor(
            a_shared, WGMMA_SWIZZLE_128B, 0, 0, 0
        )
        desc_b = S.nvvm.make_wgmma_descriptor(
            b_shared, WGMMA_SWIZZLE_128B, 0, 0, 0
        )
        acc = S.nvvm.wgmma_init_accumulator(TILE_M, TILE_N)
        acc = S.nvvm.wgmma_async(desc_a, desc_b, acc)
        S.nvvm.wgmma_store(acc, c_shared)
        S.syncthreads()

        for i in S.range(STORE_ITERS):
            idx = tid + i * THREADS_PER_CTA
            row = idx // TILE_N
            col = idx % TILE_N
            partial = c_shared[row, col]
            current = S.convert(c[block_m + row, block_n + col], S.f32)
            c[block_m + row, block_n + col] = S.convert(
                current + partial, S.bf16
            )

        S.syncthreads()


def get_hopper_device(requested_device):
    if not torch.cuda.is_available() or torch.version.cuda is None:
        raise RuntimeError("CUDA is not available.")

    device_indices = (
        [requested_device]
        if requested_device is not None
        else range(torch.cuda.device_count())
    )
    for device_idx in device_indices:
        major, _minor = torch.cuda.get_device_capability(device_idx)
        if major >= 9:
            torch.cuda.set_device(device_idx)
            torch.empty(1, device=f"cuda:{device_idx}")
            torch.cuda.synchronize(device_idx)
            return device_idx

    raise RuntimeError("No NVIDIA Hopper-or-newer GPU with WGMMA support found.")


def repeats_for_size(size):
    if size <= 1024:
        return 200
    if size <= 2048:
        return 100
    if size <= 4096:
        return 50
    if size <= 8192:
        return 20
    return 5


def validate_square_result(c, size, device):
    expected_value = torch.tensor(float(size), dtype=c.dtype, device=device)
    max_temp_elements = 16 * 1024 * 1024
    rows_per_chunk = max(1, min(size, max_temp_elements // size))
    max_diff = 0.0

    for row_start in range(0, size, rows_per_chunk):
        chunk = c[row_start : row_start + rows_per_chunk]
        if torch.all(chunk == expected_value).item():
            continue
        chunk_diff = torch.max(torch.abs(chunk.float() - float(size))).item()
        max_diff = (
            chunk_diff if chunk_diff != chunk_diff else max(max_diff, chunk_diff)
        )
        raise AssertionError(
            f"Validation failed for size {size}; max_abs_diff={max_diff}."
        )


def make_timed_launch(launch, use_graph, device):
    if not use_graph:
        return launch

    graph = torch.cuda.CUDAGraph()
    capture_stream = torch.cuda.Stream(device=device)
    try:
        with torch.cuda.graph(graph, stream=capture_stream):
            launch()
    except Exception as exc:
        raise RuntimeError(
            "CUDA graph capture failed after JIT warmup. Re-run with --no-graph "
            "to time direct launches instead."
        ) from exc
    torch.cuda.synchronize(device)
    return graph.replay


def benchmark_size(size, repeat, warmup, validate, device, use_graph):
    if size % TILE_M != 0:
        raise ValueError(f"size must be a multiple of {TILE_M}, got {size}.")

    a = torch.ones((size, size), dtype=torch.bfloat16, device=device)
    b = torch.ones((size, size), dtype=torch.bfloat16, device=device)
    c = torch.zeros((size, size), dtype=torch.bfloat16, device=device)

    grid = (size // TILE_N, size // TILE_M, 1)
    block = (THREADS_PER_CTA, 1, 1)

    def dims():
        return grid, block

    def launch():
        gemm_wgmma_bf16_square_kernel[dims](a, b, c, size)

    # First launch compiles the kernel, builds the launcher, and initializes
    # driver handles. Keep all one-time setup out of the timed region.
    c.zero_()
    launch()
    torch.cuda.synchronize(device)

    for _ in range(warmup):
        launch()
    torch.cuda.synchronize(device)

    timed_launch = make_timed_launch(launch, use_graph, device)

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    start_evt.record()
    for _ in range(repeat):
        timed_launch()
    end_evt.record()
    torch.cuda.synchronize(device)

    elapsed_ms = start_evt.elapsed_time(end_evt)
    avg_ms = elapsed_ms / repeat
    flops = 2.0 * size * size * size
    tflops = flops / (avg_ms * 1.0e9)

    if validate:
        c.zero_()
        launch()
        torch.cuda.synchronize(device)
        validate_square_result(c, size, device)

    return avg_ms, tflops


def main():
    parser = argparse.ArgumentParser(
        description="Avelang NVIDIA WGMMA BF16 square GEMM benchmark"
    )
    parser.add_argument("--sizes", nargs="+", type=int, default=DEFAULT_SIZES)
    parser.add_argument("--repeat", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--validate", action="store_true", default=False)
    parser.add_argument(
        "--no-graph",
        action="store_false",
        dest="use_graph",
        help="Time direct Python launches instead of CUDA graph replays.",
    )
    parser.set_defaults(use_graph=True)
    args = parser.parse_args()

    device_idx = get_hopper_device(args.device)
    device = torch.device(f"cuda:{device_idx}")
    name = torch.cuda.get_device_name(device_idx)
    major, minor = torch.cuda.get_device_capability(device_idx)

    print("# Avelang WGMMA BF16 square GEMM benchmark")
    print(f"selected_cuda_device,{device_idx}")
    print(f"device,{name}")
    print(f"compute_capability,{major}.{minor}")
    print(f"torch_version,{torch.__version__}")
    print(f"cuda_version,{torch.version.cuda}")
    print(f"timing_mode,{'cuda_graph' if args.use_graph else 'direct'}")
    print("M,N,K,repeats,avg_ms,tflops")

    for size in args.sizes:
        repeat = args.repeat or repeats_for_size(size)
        avg_ms, tflops = benchmark_size(
            size, repeat, args.warmup, args.validate, device, args.use_graph
        )
        print(f"{size},{size},{size},{repeat},{avg_ms:.4f},{tflops:.2f}", flush=True)


if __name__ == "__main__":
    main()
