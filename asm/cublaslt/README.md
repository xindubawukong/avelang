# cuBLASLt runtime kernel assembly dump

This directory contains the cuBLASLt BF16 square GEMM kernels observed in
`cublaslt/artifacts/bf16_square_gemm_cublaslt`.

The `nvjet_tst_*.cubin` files were captured from `cuLibraryLoadData` during the
benchmark run with `LD_PRELOAD=./asm/cublaslt/libdump_cuda_modules.so`, then
renamed to match the kernel names. The matching `nvjet_tst_*.sass` files were
generated with `cuobjdump --dump-sass` from those captured cubins.

Observed kernel mapping from `nsys`:

| Matrix size | Instances | Kernel | Captured cubin |
| --- | ---: | --- | --- |
| `1024x1024x1024` | 203 | `nvjet_tst_128x64_64x8_1x2_h_bz_NNT` | `nvjet_tst_128x64_64x8_1x2_h_bz_NNT.cubin` |
| `2048x2048x2048` | 103 | `nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NNT` | `nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NNT.cubin` |
| `4096x4096x4096` | 53 | `nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NNT` | `nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NNT.cubin` |
| `8192x8192x8192` | 23 | `nvjet_tst_320x128_64x3_1x2_h_bz_coopB_NNT` | `nvjet_tst_320x128_64x3_1x2_h_bz_coopB_NNT.cubin` |
| `16384x16384x16384` | 8 | `nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NNN` | `nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NNN.cubin` |

The instance counts are `3` warmup launches plus the benchmark repeats for each
size. The `2048` and `4096` cases use the same cuBLASLt kernel, so `nsys`
reports `156 = 103 + 53` total instances for that kernel.

`cublaslt_trace_cuda_gpu_kern_sum.csv` records the observed kernel names and
timings from `nsys`. `*.resource_usage.txt` and `*.elf_symbols.txt` are
generated from the same captured cubins.

## Reproduce

Run the full capture and disassembly flow from the repository root:

```bash
bash asm/cublaslt/reproduce.sh
```

The script runs these steps:

```bash
gcc -shared -fPIC -O2 -Wall -Wextra \
  -o asm/cublaslt/libdump_cuda_modules.so \
  asm/cublaslt/dump_cuda_modules.c \
  -ldl

nsys profile \
  --trace=cuda \
  --sample=none \
  --cpuctxsw=none \
  --backtrace=none \
  --force-overwrite=true \
  -o asm/cublaslt/cublaslt_trace \
  cublaslt/artifacts/bf16_square_gemm_cublaslt

nsys stats \
  --report cuda_gpu_kern_sum \
  --format csv \
  --force-export true \
  --force-overwrite true \
  --output asm/cublaslt/cublaslt_trace \
  asm/cublaslt/cublaslt_trace.nsys-rep

mkdir -p asm/cublaslt/runtime_modules
CUDA_MODULE_DUMP_DIR="${PWD}/asm/cublaslt/runtime_modules" \
LD_PRELOAD="${PWD}/asm/cublaslt/libdump_cuda_modules.so" \
  cublaslt/artifacts/bf16_square_gemm_cublaslt
```

After the preload run, locate each captured cubin by kernel name and disassemble
it:

```bash
for name in \
  nvjet_tst_128x64_64x8_1x2_h_bz_NNT \
  nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NNT \
  nvjet_tst_320x128_64x3_1x2_h_bz_coopB_NNT \
  nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NNN; do
  src="$(grep -a -l "${name}" asm/cublaslt/runtime_modules/*.cubin | head -n 1)"
  cp -p "${src}" "asm/cublaslt/${name}.cubin"
  /usr/local/cuda/bin/cuobjdump --dump-sass --function "${name}" \
    "asm/cublaslt/${name}.cubin" > "asm/cublaslt/${name}.sass"
  /usr/local/cuda/bin/cuobjdump --dump-elf-symbols \
    "asm/cublaslt/${name}.cubin" > "asm/cublaslt/${name}.elf_symbols.txt"
  /usr/local/cuda/bin/cuobjdump --dump-resource-usage \
    "asm/cublaslt/${name}.cubin" > "asm/cublaslt/${name}.resource_usage.txt"
done
```

The helper script removes transient `runtime_modules`, `.nsys-rep`, `.sqlite`,
and preload log files after generating the final artifacts.
