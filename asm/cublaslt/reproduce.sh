#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${ROOT}/asm/cublaslt"
BENCH="${ROOT}/cublaslt/artifacts/bf16_square_gemm_cublaslt"
CUOBJDUMP="${CUOBJDUMP:-/usr/local/cuda/bin/cuobjdump}"
NSYS="${NSYS:-nsys}"
CC="${CC:-gcc}"

KERNELS=(
  "nvjet_tst_128x64_64x8_1x2_h_bz_NNT"
  "nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NNT"
  "nvjet_tst_320x128_64x3_1x2_h_bz_coopB_NNT"
  "nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NNN"
)

if [[ ! -x "${BENCH}" ]]; then
  echo "benchmark not found or not executable: ${BENCH}" >&2
  exit 1
fi

mkdir -p "${OUT}"

"${CC}" -shared -fPIC -O2 -Wall -Wextra \
  -o "${OUT}/libdump_cuda_modules.so" \
  "${OUT}/dump_cuda_modules.c" \
  -ldl

"${NSYS}" profile \
  --trace=cuda \
  --sample=none \
  --cpuctxsw=none \
  --backtrace=none \
  --force-overwrite=true \
  -o "${OUT}/cublaslt_trace" \
  "${BENCH}" >/dev/null

"${NSYS}" stats \
  --report cuda_gpu_kern_sum \
  --format csv \
  --force-export true \
  --force-overwrite true \
  --output "${OUT}/cublaslt_trace" \
  "${OUT}/cublaslt_trace.nsys-rep" >/dev/null

runtime_dir="${OUT}/runtime_modules"
rm -rf "${runtime_dir}"
mkdir -p "${runtime_dir}"

CUDA_MODULE_DUMP_DIR="${runtime_dir}" \
LD_PRELOAD="${OUT}/libdump_cuda_modules.so" \
  "${BENCH}" >"${OUT}/preload_run.results.csv" 2>"${OUT}/preload_run.stderr.log"

for name in "${KERNELS[@]}"; do
  src="$(grep -a -l "${name}" "${runtime_dir}"/*.cubin | head -n 1 || true)"
  if [[ -z "${src}" ]]; then
    echo "could not find captured cubin for ${name}" >&2
    exit 1
  fi

  cp -p "${src}" "${OUT}/${name}.cubin"
  "${CUOBJDUMP}" --dump-sass --function "${name}" "${OUT}/${name}.cubin" \
    >"${OUT}/${name}.sass"
  "${CUOBJDUMP}" --dump-elf-symbols "${OUT}/${name}.cubin" \
    >"${OUT}/${name}.elf_symbols.txt"
  "${CUOBJDUMP}" --dump-resource-usage "${OUT}/${name}.cubin" \
    >"${OUT}/${name}.resource_usage.txt"
done

rm -rf \
  "${runtime_dir}" \
  "${OUT}/cublaslt_trace.nsys-rep" \
  "${OUT}/cublaslt_trace.sqlite" \
  "${OUT}/preload_run.results.csv" \
  "${OUT}/preload_run.stderr.log"

echo "Generated cuBLASLt runtime SASS under ${OUT}"
