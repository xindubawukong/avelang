# Dynamic MXFP4 local MoE

LLVM 24, gfx950 (MI350), BF16 input/output and dynamically quantized MXFP4
activations/weights. Stage1/Stage2 and intermediate quantization use Ave DSL.
BF16 input quantization and scale sorting call AITER as two separate kernels.

The two-stage implementation follows Petit `dev-megamoe` at
`3f64a87a5dd95f0bc01f3df92aeb05da27194e08`, the parent of the Kimi commit
`d97fb857`. The Ave module/helper organization is retained. This API always
selects the existing two-stage path; it does not add Petit's one-stage fallback.

Install the `moe` extra (`pip install -e '.[moe]'`) with a ROCm-compatible
`amd-aiter` build, or install AITER from the local checkout. AITER is imported
when preparing BF16 input; importing the compute kernels does not load it.

## Operation and profiles

The caller supplies AITER-style expert-sorted routes, with valid rows followed
by padding in each Stage1 tile. Each call quantizes BF16 input, computes W13
gate/up projections, applies activation and intermediate MXFP4 quantization,
then computes W2 and accumulates the weighted top-k contributions in BF16.
Router/top-k selection, expert sorting and communication belong to the caller.

| Benchmark profile | D | I | E | Top-k | Activation / bias |
| --- | ---: | ---: | ---: | ---: | --- |
| gptoss | 3072 | 3072 | 16 | 4 | OpenAI SwiGLU, optional BF16 bias |
| dsv3 | 7168 | 2048 | 33 | 9 | SiLU-dot, no bias |
| dsv4 | 7168 | 3072 | 49 | 7 | SiLU-dot, no bias |

Model labels live in the benchmark. Production selection uses dimensions,
activation, dtypes, expert count and top-k. This version supports SiLU-dot and
OpenAI SwiGLU with a K256 Stage2 pipeline; D and I must be multiples of 256.
`available_2stage_solutions` lists the implemented combinations for a request.

## API and ownership

```python
from avelang_kernels.amdgpu.local_moe import (
    ExpertWeights, MoeWorkspace, Routing, dynamic_mxfp4_moe, get_2stage_cfgs,
)

config = get_2stage_cfgs(
    token=x.shape[0], model_dim=7168, inter_dim=2048, expert=33, topk=9,
    activation="silu", bias_dtype="none",
)
weights = ExpertWeights.pack(w13, w2, s13, s2)
routing = Routing(sorted_ids, sorted_weights, sorted_experts, counts)
workspace = MoeWorkspace.allocate(x, routing, config)
out = dynamic_mxfp4_moe(x, weights, routing, config, workspace=workspace)
```

- `MoeSolutionId` owns the local ID codec and encoded choices. Dimensions are
  positive multiples of 64, at most 16320; dispatch applies kernel constraints.
  The common weight policy occupies bit40 and the Stage1 M32/M64 tile bit41,
  matching dev-megamoe. Stage2 has fixed geometry. Bits42–63 are reserved.
  Cached M32 profile IDs are unchanged; NT and M64 IDs use this earlier ABI.
- `MoeConfig` contains the solution, expert count and top-k. It derives geometry,
  LDS sizes and launch grids, and checks general problem invariants.
- `dispatch.py` binds complete IDs to Stage1/Stage2 factory pairs. Configuration
  construction does not imply launch support; execution resolves the complete
  combination through the same registry.
- `get_2stage_cfgs` selects a configuration and caches normalized requests.
  `solution_id` pins an explicit compatible choice. `weight_load_policy` sets
  the common policy for both stages. Conflicting explicit choices raise an error.

Activation, dtype and cache policy have separate normalization functions.
They accept known names and typed enums; dtype also accepts torch dtypes.
Arbitrary objects with a `.name` or a matching `str()` are rejected.

### Storage contract

All tensors are contiguous on one GPU. Prepare/pack weights outside repeated calls.

| Tensor | Shape / representation |
| --- | --- |
| input / output | BF16 `[M,D]` |
| raw W13 / W2 | uint8 `[E,2I,D/2]` / `[E,D,I/2]`, two FP4 values per byte |
| raw S13 / S2 | uint8 `[E,2I,D/32]` / `[E,D,I/32]`, E8M0 exponents |
| bias1 / bias2 | optional BF16 `[E,2,I]` / `[E,D]` |
| sorted IDs | int32 `[R]`, `(slot << 24) | token` |
| sorted route weights | FP32 `[R]` |
| sorted expert IDs | int32 `[R / config.stage1_tile_m]` |
| counts | device int32 `[padded route extent, M]` |

Each expert's routes are padded to `config.stage1_tile_m`. Capacity R covers
all routes and padding, and is a multiple of that M tile. Out-of-range tokens
or slots mark padding; negative expert IDs are skipped. Partial local expert
routing follows the same valid-prefix/trailing-padding contract and needs no
separate execution flag. Gate and up are separate complete projections in W13.

Native weights require N and K divisible by 256. Packed tensors preserve their
byte-container shapes. The hardware MFMA operates on K128 fragments within
these K256 blocks; the native weight/scale encoding retains those subtiles.

`IntermediateLayout` owns the intermediate size and views. Act uses
token/top-k-slot order; scales use row-major sorted-route order, with row
capacity padded to256. Input and weight scales retain their native layout;
`unsort_scales` applies to those native scales, not the intermediate workspace.
Input quantization follows AITER's upward-rounded amax/6 scale; intermediate
quantization follows Petit's distinct rounding rule.

The result aliases `workspace.out`; clone it to retain an earlier result.
Warm up before graph capture. Input and routing tensor contents may change
in-place before replay, within the allocated capacity and routing contract.
Optional biases can be omitted independently in BF16-bias configurations.

## Scheduling and implementation

The default Stage1 tile is M32/N256, where N includes both gate and up
projections. M64/N512 is also available through an explicit solution ID.
Both use four waves. Stage2 uses M32/N256/K256 and 256 persistent workers per
output-column tile. Each worker partitions the live route extent from device
counts and synchronizes before reusing its shared arena.

Automatic weight policy is non-temporal below 64 routed tokens per expert,
and cached otherwise. One policy applies to both stages.

| Responsibility | Ave files | Implementation |
| --- | --- | --- |
| Selection and encoding | `solutionid.py`, `config.py`, `dispatch.py` | Complete factory registry, shape and policy selection |
| Input preparation | `api.py` → AITER | Preallocated buffers; separate input quantization and scale sorting |
| Stage1 input/weights | `input_mxfp4.py`, `weight_mxfp4.py` | Double LDS slots, XOR layout, 16-byte act / 4-byte scale DMA |
| Stage1 compute | `stage1.py`, `activation.py` | Bulk W13 loads, full VMEM waits, existing hot-loop instruction grouping, SiLU-dot/SwiGLU |
| Intermediate storage | `intermediate_mxfp4.py`, `scale_layout.py` | Row-major intermediate scales; packed scale stores and word-load/lane-shuffle reads |
| Stage2 | `stage2.py` | Input LDS stores then W2 loads then barrier; scalar BF16 C-shuffle and branchless packed atomics |
| Workspace and execution | `api.py` | Caller-owned graph-safe buffers; clears output before Stage2 |

Stage1's compute helper produces an FP32 activation tile in LDS; its wrapper
owns route lookup and intermediate stores. Stage2's compute helper produces
weighted BF16 values in linear LDS; its wrapper owns task selection,
valid-row checks and global atomic writes. Both use weight as MFMA operand A.
Access helpers issue memory operations; the compute pipelines own waits and
barriers. The input arena is reused for output only after its consumers finish.

Stage1 skips the terminal input prefetch, retains the bulk weight-load sequence,
and fully waits before consuming the next input tile. The existing hot-loop
scheduler predates Kimi. Stage2 retains the local branch's per-K trailing
barrier, caches routing weights and output offsets in the LDS metadata tail,
and loads bias/route-weight fragments after the final MFMA. Output BF16 elements
are stored individually into linear LDS and read back as pairs for buffer
atomics, with hardware bounds discarding invalid rows.

The later native intermediate-scale format, independent stage cache policies,
four-phase W13 prefetch, partial VMEM waits, before-StoreLds W2 prefetch, packed
BF16 LDS stores, wave-uniform output skipping, output XOR layout and post-GEMM
bound snapshot are absent.

## Validation and benchmark

```bash
pytest -q test/examples/gemm/amdgpu/test_moe_*.py \
  test/examples/gemm/amdgpu/test_dynamic_mxfp4_moe.py
python benchmark/moe/bench_dynamic_mxfp4_moe.py \
  --models gptoss dsv3 dsv4 --tokens 8 128 1024 \
  --scopes stage1 stage2 compute dynamic
```

The benchmark generates synthetic inputs and fixes routing before timing.
Stage2 includes output clearing; `dynamic` also includes AITER input quantization
and scale sorting. Weight preparation and host allocation are outside the
captured graphs. Results go to stdout, or to an explicit `--output` path.
