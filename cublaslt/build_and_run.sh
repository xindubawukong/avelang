#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

SM="${SM:-80}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
NVCC="${NVCC:-${CUDA_HOME}/bin/nvcc}"
CUOBJDUMP="${CUOBJDUMP:-${CUDA_HOME}/bin/cuobjdump}"
NVDISASM="${NVDISASM:-${CUDA_HOME}/bin/nvdisasm}"

mkdir -p artifacts

COMMON_FLAGS=(
  -std=c++17
  -O3
  -lineinfo
  -I"${CUDA_HOME}/include"
)

"${NVCC}" "${COMMON_FLAGS[@]}" \
  -gencode "arch=compute_${SM},code=sm_${SM}" \
  -gencode "arch=compute_${SM},code=compute_${SM}" \
  bf16_square_gemm_cublaslt.cu \
  -lcublasLt -lcublas \
  -o artifacts/bf16_square_gemm_cublaslt

"${NVCC}" "${COMMON_FLAGS[@]}" \
  -gencode "arch=compute_${SM},code=compute_${SM}" \
  -ptx bf16_square_gemm_cublaslt.cu \
  -o artifacts/bf16_square_gemm_cublaslt.sm_${SM}.ptx

"${NVCC}" "${COMMON_FLAGS[@]}" \
  -gencode "arch=compute_${SM},code=sm_${SM}" \
  -cubin bf16_square_gemm_cublaslt.cu \
  -o artifacts/bf16_square_gemm_cublaslt.sm_${SM}.cubin

"${NVDISASM}" artifacts/bf16_square_gemm_cublaslt.sm_${SM}.cubin \
  > artifacts/bf16_square_gemm_cublaslt.sm_${SM}.sass

"${CUOBJDUMP}" --dump-sass artifacts/bf16_square_gemm_cublaslt \
  > artifacts/bf16_square_gemm_cublaslt.executable.sass

set +e
artifacts/bf16_square_gemm_cublaslt \
  > artifacts/bf16_square_gemm_cublaslt.results.csv \
  2> artifacts/bf16_square_gemm_cublaslt.stderr.log
status=$?
set -e

cat artifacts/bf16_square_gemm_cublaslt.results.csv
cat artifacts/bf16_square_gemm_cublaslt.stderr.log >&2
exit "${status}"
