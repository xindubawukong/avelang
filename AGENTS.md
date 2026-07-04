
## DSL APIs

Read the source code to understand the APIs of the avelang DSL.

## Tools

Harness to test the performance of the model:

```
HIP_VISIBLE_DEVICES=6 python3 benchmark/gemm/bench_amdgpu_gemm.py --m=512 --n=8192 --k 8192
```

## Environment

You can only read files under `/data01/home/xiangyun/avelang`.
You can only write to `/data01/home/xiangyun/avelang/python/avelang_kernels`.

Do not read any `.git` files.

## Notes

- You can run `python3 -m pytest test/examples/gemm/amdgpu/test_amdgpu_gemm.py` to test the correctness of your implementation.
- Run `python3 tools/dump_assembly.py --target-chipset gfx942 avelang_kernels.amdgpu_gemm:_gemm_pipeline_transposed_b_kernel` to dump the assembly code of the kernel.
- Use the exact function and variable names provided. Do not create new unnecessary functions.
