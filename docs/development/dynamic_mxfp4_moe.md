# Dynamic MXFP4 local MoE

LLVM 24, gfx950 (MI350), BF16 input/output and dynamically quantized MXFP4
activations/weights.
Stage1/Stage2 and their intermediate quantization use Ave DSL. BF16 input
quantization and scale sorting call AITER.

Install the `moe` extra (`pip install -e '.[moe]'`) with a ROCm-compatible
`amd-aiter` build, or install AITER from the local checkout. AITER is imported
when preparing BF16 input; importing the DSL and compute kernels does not load it.

## Operation and profiles

The caller supplies AITER-style expert-sorted routes, with valid rows followed
by padding in each Stage1 tile. Each call quantizes BF16 input,
computes W13 gate/up projections, applies activation and intermediate MXFP4
quantization, then computes W2 and combines the weighted top-k contributions.
Router/top-k selection, expert sorting, TP communication and the surrounding
latent down/up projections belong to the caller.

| Benchmark profile | D | I | E | Top-k | Activation / bias |
| --- | ---: | ---: | ---: | ---: | --- |
| gptoss | 3072 | 3072 | 16 | 4 | OpenAI SwiGLU, optional BF16 bias |
| dsv3 | 7168 | 2048 | 33 | 9 | SiLU-dot, no bias |
| dsv4 | 7168 | 3072 | 49 | 7 | SiLU-dot, no bias |
| kimi_k3 | 3584 | 384 | 896 | 16 | SiTU v2, no bias; Kimi K3 TP8 latent MoE |

Model labels live in the benchmark. Production selection uses dimensions,
activation, dtypes, expert count and top-k. This API selects two-stage kernels.
`available_2stage_solutions` lists the implemented combinations for a request.

## API and ownership

```python
from avelang_kernels.amdgpu.local_moe import (
    ExpertWeights, MoeWorkspace, Routing, dynamic_mxfp4_moe, get_2stage_cfgs,
)

config = get_2stage_cfgs(
    token=x.shape[0], model_dim=3584, inter_dim=384, expert=896, topk=16,
    activation="situ", bias_dtype="none", is_ep=False,
)
weights = ExpertWeights.pack(w13, w2, s13, s2)
routing = Routing(sorted_ids, sorted_weights, sorted_experts, counts)
workspace = MoeWorkspace.allocate(x, routing, config)
out = dynamic_mxfp4_moe(x, weights, routing, config, workspace=workspace)
```

- `MoeSolutionId` owns the local ID codec and encoded choices. Dimensions are
  positive multiples of 64, at most 16320. Stage1/Stage2 cache policies occupy
  bits 40/41; Stage1 tile bits are 42/44/46, Stage2 tile bits 43/45. Bits 47–63
  are reserved. Activation code 2 is SiTU v2.
- `MoeConfig` contains the solution, expert count, top-k and `use_route_reduce`.
  Other fields derive from these. It checks general problem invariants.
- `dispatch.py` binds complete IDs to Stage1/Stage2 factory pairs. Configuration
  construction does not imply launch support; every execution entry resolves
  the complete combination. There is no duplicated per-field support whitelist.
- `get_2stage_cfgs` selects a configuration and caches normalized requests.
  `solution_id` pins an explicit compatible choice. `weight_load_policy` sets
  both stages; `stage1_weight_load_policy` and `stage2_weight_load_policy` allow
  separate overrides. Conflicting explicit choices raise an error.

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
or slots mark padding; negative expert IDs are skipped. Gate and up are stored
as separate complete projections in W13.

Native weight act needs K divisible by 128. Scale columns are padded to
K256 tiles with neutral exponent 127; Kimi W2 keeps its three K128 act
tiles and pads only scales from 12 to 16 columns. Packed tensors preserve their
byte-container shape except for this scale padding.

`IntermediateLayout` in `intermediate_mxfp4.py` owns the intermediate size/views. K256 Stage2 reads act
in token/top-k-slot order; K128 reads sorted route order. **Both use native
M32/K256-swizzled intermediate scales**, with row capacity padded to 256.
`unsort_scales` decodes them for inspection. Input quantization follows AITER's
upward-rounded amax/6 scale; intermediate quantization follows Petit's distinct
rounding rule.

The result aliases `workspace.out`; clone it to retain an earlier result.
Warm up before graph capture. Input and routing tensor contents may change
in-place before replay, within the allocated capacity and routing contract.
Optional biases can be omitted independently in BF16-bias configurations.

### Kimi dispatch and route reduction

| Token count | Stage1 M / combined N / K groups | Stage2 M / N / K | S1 weights | S2 weights |
| --- | --- | --- | --- | --- |
| 0–16 | 32 / 128 / 2 | 32 / 256 / 128 | NT | cached |
| 17–1024 | 32 / 128 / 2 | 32 / 256 / 128 | NT | NT |
| 1025–2047 | 32 / 128 / 2 | 32 / 256 / 128 | NT | cached |
| 2048 | 64 / 256 / 1 | 64 / 256 / 128 | NT | cached |
| 2049+ | 64 / 256 / 1 | 64 / 256 / 128 | cached | cached |

Combined N counts gate and up together. At M4096+, Kimi Stage1 uses grouped
workgroup mapping. K128 Stage2 uses a spatial grid bounded by the routing
capacity and worst-case active expert padding, then maps against device counts.

At M8192+, complete local Kimi routing uses BF16 `[M,topk,D]` route output,
followed by FP32 top-k reduction and one BF16 rounding. It avoids output clear
and packed atomics. **Every token/slot must occur exactly once and all selected
experts must be local.** Use `is_ep=True` for partial routing; it disables this
path. `use_route_reduce=False` explicitly selects atomics. Partial local routing
must retain valid prefixes and trailing padding within each tile. Metadata validation
does not copy device counts/IDs to the host.

## Implementation

| Responsibility | Ave files | Implementation / review finding |
| --- | --- | --- |
| Problem selection and encoding | `solutionid.py`, `config.py`, `dispatch.py` | Independent stage policies, scattered tile bits, Kimi thresholds; executable factory catalog |
| Offline storage | `api.py`, `scale_layout.py` | `ExpertWeights.pack` owns weight/bias packing; K128 act and padded scale support; intermediate scales migrated for old profiles too |
| Input preparation | `api.py` → AITER | Preallocated buffers; fused small-batch / split large-batch dispatch |
| Stage1 input/weights | `input_mxfp4.py`, `weight_mxfp4.py` | Shared M32/M64/K2 geometry, double LDS slots, 16-byte act / 4-byte scale DMA, cached invariant route offsets; act DMA before scale DMA |
| Stage1 computation | `stage1.py`, `activation.py` | Four-phase weight prefetch and MFMA clusters, priority control, partial vmcnt waits, K2 merge in LDS, SiTU exponent precomputation and per-float4 finalization/LDS write overlap, with the same fused denominator rounding |
| Intermediate storage | `intermediate_mxfp4.py`, `scale_layout.py` | Allocation layout, quantization/stores and Stage2 reads; NT act stores and cached byte scale stores; independent act/scale rows |
| K128 Stage2 | `stage2.py` | Full I resident in compact swizzled LDS, activation as MFMA A, double weight registers, K256 scale reuse across K128 tiles, interleaved last-tile C-shuffle with float2 multiply/BF16 conversion and route-ID prefetch |
| K256 Stage2 | `stage2.py` | Direct native scale loads, weights before input LDS write, fewer barriers, packed valid-row C-shuffle, per-wave writeback |
| Spatial mapping | `workgroup.py` | One grouped mapping helper for both stages, including partial groups |
| Large-batch reduction | `route_reduce.py` | Preload routes, FP32 accumulation, token-relative buffer descriptors and large-address handling |
| Execution / workspace | `api.py` | Graph-safe caller-owned buffers; route workspace checked before launches; stage grids centralized in config |
| DSL extension | `amdgpu_mxfp4.cc`, `amdgpu_module.cc`, Python intrinsic declaration | `raw_buffer_store_u8` for native scale bytes; neighboring bytes and descriptor bounds tested |

### Ownership and device interfaces

Local MoE owns the shared computation and storage implementations. MegaMoE
imports these modules; local MoE has no dependency on MegaMoE, its scheduler or
symmetric heap.

- `stage1.py` contains `make_stage1_compute`, the local routing/writeback
  wrapper and the `make_stage1` entry point. The shared computation ends with
  an FP32 activation tile in LDS. Compute, input and weight helpers read the
  selected `MoeConfig` or `MegaMoeConfig` directly through their common
  `compute_hidden`, `intermediate`, `activation`, `bias` and `stage1_*` fields.
  Each configuration owns its geometry and LDS size calculations.
- `stage2.py` contains separate `make_stage2_compute_k256` and
  `make_stage2_compute_k128` factories with their fragment epilogues, followed
  by local task selection, routing and global output writes.
- `make_stage1`/`make_stage2` resolve a solution and cache the resulting kernel.
  The registered `make_stage1_kernel`/`make_stage2_kernel` factories construct
  the local kernels. MegaMoE reuses the compute factories from those same files.
- `input_mxfp4.py` owns Stage1 routed act/scale access. Mega supplies its
  own `input_mxfp4_packed.py` adapter. `weight_mxfp4.py` owns expert/tile
  descriptors and weight loads; offline packing lives beside `ExpertWeights`
  in `api.py`.
- Stage1 and the separate K256/K128 Stage2 pipelines explicitly order prefetch,
  `s_waitcnt`, barriers and MFMA. Access helpers do not insert pipeline barriers.
  K256 keeps its short weight-load issue group inside the K loop: extracting
  that group changed register allocation and caused a measured DSv4 EP8
  regression. Expert/tile descriptors remain shared in `weight_mxfp4.py`.
- `intermediate_mxfp4.py` owns `IntermediateLayout`, float4 quantization/stores
  and the K256/K128 Stage2 input readers, keeping producer and consumer together.
  Its caller supplies **independent act and scale rows**: ordinary local
  act uses token/slot order while scales use sorted route order; Kimi uses
  sorted rows for both, and Mega uses expert-pool rows. Shared scale shapes,
  byte offsets and host decoding are defined together in `scale_layout.py`. Validity
  and readiness publication remain in the wrappers.
- Stage2's private epilogue helpers live next to their respective pipelines.
  K256 retains accumulator traversal and row-major LDS
  coordinates in its pipeline: passing the whole accumulator to a helper raised
  GPT-OSS VGPR use from 128 to 132 and measurably slowed Stage2. The pair interface
  restores 128 VGPRs. K128 calls its writer between the final MFMA clusters,
  preserving the overlap.
- LDS is reused only at pipeline-controlled barriers: input stages become
  W13 partial/activation storage or W2 output storage. Each factory documents
  its fragment dimensions, units and synchronization contract.

Stage1 variants share one pipeline. K128 M32/M64 Stage2 share one factory;
K256 remains a separate schedule because its input and MFMA operand order differ.
The K256 writeback checks each output row's validity. Its output-size bound
is read once immediately before the writeback loop. Reading
`counts[1]` inside every row check generated repeated global loads and VMEM waits
between atomics. The late snapshot removes seven such loads/waits without
extending the bound's lifetime across the GEMM or changing the 128-VGPR GPT-OSS
kernel.

The Stage2 C-shuffle uses shape-preserving vector `al.convert`: a float2
multiply is followed by one bf16x2 conversion, then low/high-half LDS stores.
The generic conversion implementation is in `builtin_module.cc` and
`type_promotion.cc`; rounding, widening, unsigned inputs and f16/bf16 format
conversion are covered by `test_vector_convert.py`.

## Validation and benchmark

```bash
pytest -q test/examples/gemm/amdgpu/test_moe_*.py \
  test/examples/gemm/amdgpu/test_dynamic_mxfp4_moe.py
python benchmark/moe/bench_dynamic_mxfp4_moe.py \
  --models gptoss dsv3 dsv4 kimi_k3 --tokens 8 128 1024 \
  --scopes stage1 stage2 compute dynamic
```

The benchmark generates synthetic inputs and fixes routing before timing.
Stage2 includes output clearing or route reduction; `dynamic` also includes
AITER input quantization and scale sorting. Weight preparation and host
allocation are outside the captured graphs. Results go to stdout, or to an
explicit `--output` path.

External Petit comparisons, checkpoint replay and historical reports live in
`~/compare/avelang-petit/`. They are not required by these tests or benchmarks.
