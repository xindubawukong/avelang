## Goal

Optimize the avelang kernel in `python/avelang_kernels/amdgpu_gemm.py`.

This prompt is incremental. Sections are separated by `---`.
Apply one section in each iteration, and only introduce an optimization when the section introduces the needed layout/invariant.
Run the test and report the performance after each section.

The intended progression is:

1. First write a simple correct avelang kernel.
2. Then add block tiling and shared memory.
3. Then introduce MFMA and its layout invariants.
4. Then optimize writeback, global memory, pipelining, and LDS layout.

Do not start with MFMA. Do not assume MFMA accumulator layout until the MFMA section introduces it.


## Kernel Settings

- `WARP_SIZE = 64`
- `NUM_WARPS = 4`
- `BLOCK_SIZE = WARP_SIZE * NUM_WARPS`
- `GROUP_M = 128`, `GROUP_N = 128`, `GROUP_K = 64`
- `WARP_PER_ROW = 2`, `WARP_PER_COL = 2`
- `WARP_MAT_M = GROUP_M // WARP_PER_ROW`, `WARP_MAT_N = GROUP_N // WARP_PER_COL`
- `M_TILES_PER_WARP = WARP_MAT_M // 16`, `N_TILES_PER_WARP = WARP_MAT_N // 16`
- `VEC_SIZE = 8`: number of bf16 elements per 16-byte vector
- `BF16_BYTES = 2`

Use `grid_size = m_groups * n_groups`.
Use `block_size = BLOCK_SIZE`.

You can get the following variables:

- `tid = al.thread_id(0)`
- `wid = tid // WARP_SIZE`
- `wtid = tid % WARP_SIZE`
- `warp_row = wid // WARP_PER_COL`
- `warp_col = wid % WARP_PER_COL`

A is row-major `(m, k)`.
B is transposed and stored row-major as `(n, k)`.
C is row-major `(m, n)`.
The result is `C = A @ B.T`.

Treat `(m, n, k)` as input variables instead of hardcoding sizes.


## Overall Flow

Each block computes a `GROUP_M x GROUP_N` tile of output C.
The 4 warps are interpreted as a 2 x 2 warp grid, where each warp later works on a 64x64 tile in C.

At every optimization step, preserve the same mathematical dataflow:

`A(row, kk) * B(col, kk)` accumulated over `kk`, then stored to `C(row, col)`.

Later optimizations may change the layout and schedule, but they must not change the math.


## Baseline Kernel

First implement the simplest correct avelang kernel.
This baseline is not expected to be fast.

Use an avelang kernel, not PyTorch fallback.
At this stage, keep the implementation intentionally simple: no shared memory, no MFMA, no raw buffer operations, and no validation logic.
Those optimizations are introduced in later sections.

Each block computes one output tile.
Within a block, distribute the `GROUP_M * GROUP_N` output elements across the `BLOCK_SIZE` threads.
Each thread computes one or more C elements by looping over K in ordinary scalar code:

- compute the output `row` and `col`
- initialize an f32 accumulator
- loop over `kk in [0, k)`
- accumulate `float(A[row, kk]) * float(B[col, kk])`
- cast the final accumulator to bf16 and store C

Use this baseline only to establish correctness and launch structure.
All non-trivial kernel helpers must be top-level `@avelang.jit` functions.


---


# Optimization 1: Load Tiles to LDS

Now move from the naive scalar kernel to block-level K tiling.
This section only introduces the global-memory to shared-memory path.
Do not use MFMA yet.
Use ordinary scalar loops from LDS for compute.

Create:

- `shm_a` for a `GROUP_M x GROUP_K` tile of A
- `shm_b` for a `GROUP_N x GROUP_K` tile of B
- `reg_a, reg_b` as temporary register buffers for global-to-shared loads

Add:

- `_load_global`: load a tile of A/B from global memory to registers
- `_store_shm`: store the register tile to shared memory
- `_write_results`: write scalar accumulated results to C

The K loop iterates over `GROUP_K` chunks:

1. `_load_global(A, reg_a)`, `_load_global(B, reg_b)`
2. `_store_shm(A, reg_a)`, `_store_shm(B, reg_b)`
3. `al.syncthreads()`
4. compute this K chunk using ordinary scalar dot products from shared memory
5. `al.syncthreads()`

After the K loop, call `_write_results`.

The purpose of this step is to verify the tile ownership and LDS layout before adding MFMA.
It is acceptable if this version is slow.


---


# Optimization 2: MFMA and LDS Pattern

Now replace the scalar LDS compute with MFMA.
This is the first section where MFMA layout matters.

Create:

- `data_a = al.make_local((M_TILES_PER_WARP, 2, 4), al.u32)`
- `data_b = al.make_local((N_TILES_PER_WARP, 2, 4), al.u32)`
- `acc = al.make_local((M_TILES_PER_WARP, N_TILES_PER_WARP, 4), al.f32)`

Initialize all `acc` values to f32 zero.

Add:

- `_load_shm_to_regs`: load A/B fragments from shared memory to `data_a/data_b`
- `_matmul_from_regs`: feed `data_a/data_b` fragments to MFMA and accumulate into `acc`

The K loop now keeps the same global-to-LDS path from Optimization 1, but changes the compute part:

1. `_load_global(A, reg_a)`, `_load_global(B, reg_b)`
2. `_store_shm(A, reg_a)`, `_store_shm(B, reg_b)`
3. `al.syncthreads()`
4. `_load_shm_to_regs(A, data_a)`, `_load_shm_to_regs(B, data_b)`
5. `_matmul_from_regs(data_a, data_b, acc)`
6. `al.syncthreads()`

After the K loop, `_write_results` must store the MFMA accumulator results.

## MFMA

Use:

`acc = al.amdgpu.mfma_16x16x16_bf16_f32(data_a, data_b, acc)`

Do not change the order of `data_a` and `data_b`.
Never call MFMA as `mfma(data_b, data_a, acc)`.
If writeback is inconvenient, fix `_write_results`; do not swap operands.

MFMA invariants for `al.amdgpu.mfma_16x16x16_bf16_f32`:

For A, `i in [0,16)`, `kk in [0,16)`:

- `A(i, kk) -> (wtid = i + (kk // 4) * 16, element = kk % 4)`

For B, `kk in [0,16)`, `j in [0,16)`:

- `B(kk, j) -> (wtid = j + (kk // 4) * 16, element = kk % 4)`

Accumulator invariants for C, for each `wtid in [0,64)` and `acc_idx in [0,4)`:

- `col = tile_col_base + (wtid % 16)`
- `row = tile_row_base + 4 * (wtid // 16) + acc_idx`


## LDS to MFMA

For `GROUP_K = 64`, each K tile has two 32-column batches:

- `batch_id = 0`: columns 0-31
- `batch_id = 1`: columns 32-63

In `_load_shm_to_regs`, each thread reads one 16-byte fragment from each of `M_TILES_PER_WARP` or `N_TILES_PER_WARP` contiguous rows, for both batches.

The intended LDS-to-MFMA swizzle is:

- rows are selected by `wtid % 16`
- K-vector groups are selected by `wtid // 16`
- each row contributes two 16-byte fragments, one for each 32-column batch

Concretely:

- A rows start at `row_base + (wtid % 16) * M_TILES_PER_WARP`
- B rows start at `row_base + (wtid % 16) * N_TILES_PER_WARP`
- batch columns start at `batch_id * 32 + (wtid // 16) * VEC_SIZE`

When reusing `_load_shm_to_regs` for A and B, use the correct tile count for each operand: `M_TILES_PER_WARP` for A and `N_TILES_PER_WARP` for B.

Treat each 16-byte fragment as `(4, al.u32)` and reinterpret it as two natural `(4, al.bf16)` halves.
Feed both halves into MFMA in natural order.
The intended effect is a cooperative 16x16x32 accumulation from two natural MFMA steps.


---


# Optimization 3: Write Back

After the results are stored in local `acc`, write `_write_results` to directly store `acc` to global memory with vectorized stores.
Do not involve shared memory in `_write_results`.
Do not use `al.shuffle`.
Do not change `_matmul_from_regs`.
Do not change the order of `data_a` and `data_b`.

`_write_results` is about undoing the layouts introduced by the previous section:

- the MFMA accumulator layout
- the row swizzle introduced by `_load_shm_to_regs`

For vectorized stores, pack neighboring output columns, not neighboring output rows.
In this kernel, neighboring columns come from different `n_tile` entries of `acc` with the same `m_tile` and `acc_idx`.

Note that the output of MFMA is f32, and we need to cast it back to bf16.
For `_write_results`, you can refer to the following code snippet to convert two f32 to two bf16:

```python
lo = al.bitcast(data[0], al.u32)
hi = al.bitcast(data[1], al.u32)
res = al.amdgpu.perm(hi, lo, 0x07060302)
```

The `res` is type `al.u32`, which can be reinterpreted as two bf16 values.
Store two packed `al.u32` values with `al.amdgpu.raw_buffer_store_x2`.


---


# Optimization 4: Global Memory Operations

Load data from global memory using vectorized loads:

`al.amdgpu.raw_buffer_load_x4(rsrc, vindex, soffset, aux)`

Write results back to global memory using vectorized stores:

`al.amdgpu.raw_buffer_store_x2(vdata, rsrc, vindex, soffset, aux)`

Create `A_rsrc`, `B_rsrc`, `C_rsrc` before the main loop and reuse them for data load/store.
Prefer full-tensor resources and put block offsets in `soffset`.

When using buffer load/store, put thread-level offsets in `vindex` and block/warp-level offsets in `soffset` where possible.

Critical idea for `_load_global`:

Vectorization per thread is not enough.
The same static load instruction must also be coalesced across lanes in the wave.

Therefore, distribute the tile so that for a fixed `load_idx`, adjacent lanes load adjacent 16-byte chunks.
The thread-to-tile mapping should be based on:

- `elem_base = (load_idx * BLOCK_SIZE + tid) * VEC_SIZE`

Do not use:

- `elem_base = (tid * loads_per_thread + load_idx) * VEC_SIZE`

The second form makes each thread locally contiguous, but makes the wave access strided addresses and is much slower.

Use the same logical `elem_base` in `_store_shm`.
The data loaded by a thread must be stored to the shared-memory row/column corresponding to that same `elem_base`.
Do not invent a different thread-to-LDS mapping in `_store_shm`.

Utilize range in `al.amdgpu.make_rsrc` to remove explicit branches guarding OOB access.
The range is in bytes.
When range is set, `raw_buffer_load_*` returns 0 for OOB elements, and `raw_buffer_store_*` discards OOB writes.
For partial C tiles, still make sure a write past the logical end of a row does not land in the next row.


---


# Optimization 5: Pipelining

Recall that for `GROUP_K = 64`, there are two batches for each tile, each working on 32 columns.

One k-step calls the following functions in this order:

1. `_load_shm_to_regs A/B` for k-tile `k_idx`, `batch_id=1`
2. `_matmul_from_regs` for k-tile `k_idx`, `batch_id=0`
3. `al.syncthreads()`
4. `_store_shm A/B` for k-tile `k_idx + 1`
5. `_load_global A/B` for k-tile `k_idx + 2`
6. `al.syncthreads()`
7. `_load_shm_to_regs A/B` for k-tile `k_idx + 1`, `batch_id=0`
8. `_matmul_from_regs` for k-tile `k_idx`, `batch_id=1`

The order of steps 4 and 5 matters.
Use single buffering for the registers that store data from global memory.
First store the existing `reg_a/reg_b` contents to shared memory, then load the next tile from global memory into the same registers.

Unroll the K-loop by 2 to minimize branching, where there are two k-steps in the loop body.
Then `k_idx` should loop from `0` to `k_total - 3`.
You can assume:

- `k_total = k // GROUP_K`
- `k_total >= 3`
- `k_total % 2 == 0`

Do not involve branching inside the main loop body.

Use single buffering for shared memory: there is only one shared memory buffer for A and one for B.

Prologue idea:

1. Load k-tile 0 from global to `reg_a/reg_b`
2. Store k-tile 0 from `reg_a/reg_b` to shared memory
3. `al.syncthreads()`
4. Load k-tile 0, `batch_id=0`, from shared memory to `data_a/data_b`
5. Load k-tile 1 from global to `reg_a/reg_b`

This prologue shortens the lifetime of `reg_a/reg_b` compared with loading tile 1 before the first LDS read.


---


# Optimization 6: Shared Memory Padding

To mitigate bank conflicts, add padding in the shared memory layout.
For every 4 rows of A/B in shared memory, add 32 bytes of padding.

The shared-memory layout should still be logically row-major in `(row, K)`.
Padding changes the physical stride every 4 rows; it should not change which logical matrix element each thread owns.

Use a layout that can store and load 16-byte chunks as `(4, al.u32)`.
The exact constants should express:

- number of 16-byte vectors per logical row
- number of padding vectors per 4-row group
- number of vectors per padded 4-row group
- total number of vectors in the shared-memory tile


---


# Optimization 7: Shared Memory Operations

In `_store_shm` and `_load_shm_to_regs`, use vectorized store/load in 16-byte chunks: `(4, al.u32)`, 8 bf16 elements per chunk.

For `_store_shm`, preserve the global-load ownership:

- use the same `elem_base` mapping as `_load_global`
- derive `tile_row` and `tile_col` from that `elem_base`
- convert `(tile_row, tile_col)` into the padded shared-memory address
- store the 16-byte register fragment there

This is the key correctness/performance point:
global loads are optimized by wave-coalesced ownership, and LDS stores must respect that same ownership.

For `_load_shm_to_regs`, use the MFMA-oriented ownership:

- row ownership comes from `wtid % 16`
- K-vector ownership comes from `wtid // 16`
- batch ownership selects columns 0-31 or 32-63
- loop over contiguous rows with base + affine loop-index offsets

Prefer base + affine loop-index offsets over mutating running indices, so AMD codegen can fold LDS offsets into the instruction encoding.


---


# Optimization 8: Instruction Scheduling

Add a JIT helper `_hot_loop_scheduler()` for the GEMM kernel hot loop.
Call `_hot_loop_scheduler()` per unrolled main-loop iteration at the end of the unrolled loop body.
Do not move these calls before the per-step LDS reads or MFMA calls.

Target hardware:
- AMD gfx942
- The kernel hot loop contains MFMA compute, LDS reads, LDS writes, raw buffer global loads, and workgroup barriers.
- Use only `al.amdgpu.sched_group_barrier(mask, count, 0)` scheduling hints. Keep the helper semantic-free: it should only emit scheduling hints.

High-level scheduling idea:
The hot loop has three phases:
- Prologue-like LDS-read phase
- Main memory/compute pipeline phase
- Epilogue-like LDS-read phase

For the first phase, emit several groups of: `1 * DS_READ + 2 * MFMA`.

For the middle phase, emit several groups matching the memory pipeline:`1 * DS_WRITE + 1 * MFMA + 1 * BUFFER_LOAD + 3 * MFMA`.

For the final phase, again emit several groups of `1 * DS_READ + 2 * MFMA`.

Tuning workflow:
1. First implement a reasonable initial scheduler using the above pattern.
2. Run `python3 tools/dump_assembly.py --target-chipset gfx942 avelang_kernels.amdgpu_gemm:_gemm_pipeline_transposed_b_kernel >temp.s`.
3. Inspect the generated assembly in `temp.s` and count the actual hot-loop instruction pattern:
  - number and placement of LDS reads
  - number and placement of LDS writes
  - number and placement of raw buffer loads
4. Adjust `_hot_loop_scheduler()` so the scheduler hints match the real instruction stream.
  - Treat `DS_READ`, `DS_WRITE`, and `BUFFER_LOAD` counts/ordering as fixed anchors.

Mask constants:
- `SCHED_MASK_MFMA = 0x8`
- `SCHED_MASK_BUFFER_LOAD = 0x20`
- `SCHED_MASK_DS_READ = 0x100`
- `SCHED_MASK_DS_WRITE = 0x200`


---


# Optimization 9: WGM Mapping

Replace the direct row-major `block_id(0) -> (group_m, group_n)` mapping with an XCC-aware WGM mapping.
This optimization only changes which output tile each block owns; it must not change the per-tile compute, LDS layout, pipelining, instruction scheduling, or writeback.

Use these constants:

- `NUM_XCCS = 8`
- `CUS_PER_XCC = 38`
- `CU_SLOTS = NUM_XCCS * CUS_PER_XCC`
- `WGM_MAPPING_M_TILES = 32`

The idea has two layers:

1. XCC distribution: for each full wave of `CU_SLOTS` blocks, assign consecutive logical blocks across XCCs first, then across CUs inside each XCC. This avoids filling one XCC before using the others.
2. WGM locality: after the XCC remap, keep each scheduling region local in M by processing at most `WGM_MAPPING_M_TILES` M tiles before advancing through N.

For the tail after all full waves, keep the same distribution idea but only use as many active XCCs as there are remaining groups, then split the remaining groups as evenly as possible.

After producing `remapped_id`, convert it to `(group_m, group_n)` by grouping M tiles in chunks of `WGM_MAPPING_M_TILES`; the last M chunk may be smaller.

Use this mapping helper in the kernel before computing `row_base` and `col_base`.


---


# Verification

Run correctness after each section:

```bash
PYTHONPATH=python python3 -m pytest test/examples/gemm/amdgpu/test_amdgpu_gemm.py
```

Run the benchmark after optimization sections:

```bash
HIP_VISIBLE_DEVICES=6 PYTHONPATH=python python3 benchmark/gemm/bench_amdgpu_gemm.py --m=512 --n=8192 --k 8192 --validate
```

Dump assembly when investigating performance:

```bash
PATH=/opt/rocm/llvm/bin:$PATH PYTHONPATH=python \
python3 tools/dump_assembly.py --target-chipset gfx942 \
avelang_kernels.amdgpu_gemm:_gemm_pipeline_transposed_b_kernel
```

Sanity checks for the final optimized kernel:

- MFMA calls must use `data_a, data_b`, never swapped.
- `_load_global` must use wave-coalesced ownership across lanes for each `load_idx`.
- `_store_shm` must preserve the same ownership as `_load_global`.
- `_write_results` must pack neighboring output columns, not neighboring output rows.
- `_write_results` must not use shared memory.
- `_write_results` must not use `al.shuffle`.
