# cuBLASLt BF16 Square GEMM Results

Benchmark source: `bf16_square_gemm_cublaslt.cu`

Build/run command:

```bash
cd /workspace/avelang/cublaslt
SM=90a ./build_and_run.sh
```

The benchmark evaluates `D = alpha * A * B + beta * C` through
`cublasLtMatmul` with BF16 A/B/C/D matrices and FP32 accumulation. Effective
throughput is computed as `2*M*N*K / avg_time`.

Generated artifacts for this run:

- `artifacts/bf16_square_gemm_cublaslt`
- `artifacts/bf16_square_gemm_cublaslt.sm_90a.ptx`
- `artifacts/bf16_square_gemm_cublaslt.sm_90a.cubin`
- `artifacts/bf16_square_gemm_cublaslt.sm_90a.sass`
- `artifacts/bf16_square_gemm_cublaslt.executable.sass`
- `artifacts/bf16_square_gemm_cublaslt.results.csv`

Note: the saved PTX/SASS correspond to this CUDA benchmark translation unit,
including the BF16 initialization kernel. The GEMM implementation itself is
selected from the NVIDIA cuBLASLt library at runtime.

## Environment

- Date: 2026-06-23
- Host path: `/workspace/avelang/cublaslt`
- CUDA compiler: CUDA 13.0, `nvcc` V13.0.88
- CUDA runtime version: 13000
- cuBLAS version: 13.0.2
- Selected CUDA device: 2
- GPU: NVIDIA H100 PCIe
- Compute capability: 9.0

## Results

| M | N | K | Repeats | Avg time (ms) | Effective TFLOPS | Algo workspace bytes |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 1024 | 1024 | 200 | 0.0087 | 247.19 | 0 |
| 2048 | 2048 | 2048 | 100 | 0.0331 | 519.17 | 0 |
| 4096 | 4096 | 4096 | 50 | 0.2336 | 588.45 | 4 |
| 8192 | 8192 | 8192 | 20 | 1.5882 | 692.32 | 4 |
| 16384 | 16384 | 16384 | 5 | 16.0167 | 549.18 | 4 |

## Captured Output

`artifacts/bf16_square_gemm_cublaslt.results.csv`:

```text
# cuBLASLt BF16 square GEMM benchmark
selected_cuda_device,2
device,NVIDIA H100 PCIe
compute_capability,9.0
cuda_runtime_version,13000
cublas_version,13.0.2
M,N,K,repeats,avg_ms,tflops,algo_workspace_bytes
1024,1024,1024,200,0.0087,247.19,0
2048,2048,2048,100,0.0331,519.17,0
4096,4096,4096,50,0.2336,588.45,4
8192,8192,8192,20,1.5882,692.32,4
16384,16384,16384,5,16.0167,549.18,4
```

## Avelang Results

// TODO