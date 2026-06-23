#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"

KERNEL="gemm_wgmma_bf16_square_kernel"
SOURCE="benchmark/gemm/bench_nvidia_wgmma_bf16_square.py:${KERNEL}"
OUT_DIR="asm/avelang"
TARGET_TRIPLE="nvptx64-nvidia-cuda"
TARGET_CHIPSET="sm_90a"
SIZES=(1024 2048 4096 8192 16384)

mkdir -p "${OUT_DIR}"

for size in "${SIZES[@]}"; do
  base="${OUT_DIR}/${KERNEL}.${TARGET_CHIPSET}.size_${size}"
  constexprs="[{\"name\":\"size\",\"type\":\"i32\",\"value\":${size}},{\"name\":\"WGMMA_SWIZZLE_128B\",\"type\":\"i32\",\"value\":3},{\"name\":\"TILE_M\",\"type\":\"i32\",\"value\":64},{\"name\":\"TILE_N\",\"type\":\"i32\",\"value\":64},{\"name\":\"TILE_K\",\"type\":\"i32\",\"value\":64},{\"name\":\"THREADS_PER_CTA\",\"type\":\"i32\",\"value\":128},{\"name\":\"BF16_BYTES\",\"type\":\"i32\",\"value\":2},{\"name\":\"CP_ASYNC_BYTES\",\"type\":\"i32\",\"value\":16},{\"name\":\"BF16_PER_CP_ASYNC\",\"type\":\"i32\",\"value\":8},{\"name\":\"CP_ASYNC_CHUNKS_PER_ROW\",\"type\":\"i32\",\"value\":8},{\"name\":\"CP_ASYNC_CHUNKS\",\"type\":\"i32\",\"value\":512},{\"name\":\"STORE_ITERS\",\"type\":\"i32\",\"value\":32}]"

  python tools/dump_assembly.py "${SOURCE}" \
    --target-triple "${TARGET_TRIPLE}" \
    --target-chipset "${TARGET_CHIPSET}" \
    --constexprs-json "${constexprs}" \
    -o "${base}.ptx"

  ptxas -arch="${TARGET_CHIPSET}" -O2 -o "${base}.cubin" "${base}.ptx"
  nvdisasm "${base}.cubin" > "${base}.sass"
done

