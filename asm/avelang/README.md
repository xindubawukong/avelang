# Avelang WGMMA BF16 Square GEMM Assembly Dumps

Generate the PTX, cubin, and SASS dumps with:

```bash
./asm/avelang/generate_gemm_wgmma_bf16_square_kernel_asm.sh
```

The script dumps `gemm_wgmma_bf16_square_kernel` from:

```text
benchmark/gemm/bench_nvidia_wgmma_bf16_square.py
```

It emits one specialization per benchmark default size because `size` is an
`S.constexpr` parameter:

```text
1024 2048 4096 8192 16384
```

Each specialization produces:

```text
gemm_wgmma_bf16_square_kernel.sm_90a.size_<size>.ptx
gemm_wgmma_bf16_square_kernel.sm_90a.size_<size>.cubin
gemm_wgmma_bf16_square_kernel.sm_90a.size_<size>.sass
```

The generation pipeline is:

```text
tools/dump_assembly.py -> PTX
ptxas                 -> cubin
nvdisasm              -> SASS
```

The SASS file is the final machine-code disassembly to inspect for instructions
such as `HGMMA.64x64x16.F32.BF16`.

