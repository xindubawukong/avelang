# AMDGPU MegaMoE

`avelang_kernels.amdgpu.mega_moe` ports Petit's intra-node, dynamic MXFP4
MegaMoE to AveLang on LLVM 24 / gfx950.

## Supported configurations

| Global experts | Top-k | Logical hidden | Compute hidden | Intermediate | Activation / bias | EP |
| --- | --- | --- | --- | --- | --- | --- |
| 32 | 4 | 2880 | 3072 | 3072 | OpenAI SwiGLU / optional BF16 | 1, 2, 4, 8 |
| 128 | 4 | 2880 | 3072 | 3072 | OpenAI SwiGLU / optional BF16 | 8 |
| 256 | 8 | 7168 | 7168 | 2048 | SiLU / none | 8 |
| 384 | 6 | 7168 | 7168 | 3072 | SiLU / none | 8 |

Capacity is 1024 input tokens per rank. Every rank participates, including ranks
with zero tokens. Each token selects distinct global expert IDs in `[0, E)`.
Experts are sharded contiguously in process-group rank order. All group members
must be on the same node, with one process per GPU.

`MegaMoeSolutionId` owns the encoded topology, dimensions and instruction choices.
Its common low 24 bits share definitions with local MoE. The remaining bits follow
Petit's separate MegaMoE codec, including the producer geometry. `MegaMoeConfig`
provides derived layout/launch geometry; `get_2stage_cfgs` resolves complete IDs to
actual kernel factories. Encoding an ID does not imply that a kernel is registered.

## API

Select the local GPU and initialize `torch.distributed` before allocating a
multi-rank workspace. EP1 can run without initializing a process group.

```python
import torch
from avelang_kernels.amdgpu.mega_moe import (
    MegaMoeWorkspace, dynamic_mxfp4_mega_moe, get_2stage_cfgs,
    pack_expert_weights,
)

config = get_2stage_cfgs(
    8, world_size=8, experts=128, topk=4, hidden=2880, intermediate=3072,
    activation="swiglu", bias_dtype="bf16",
)
workspace = MegaMoeWorkspace.allocate(config)
# Row-major packed FP4 bytes, row-major E8M0 scales, and optional BF16 biases
# for this rank's experts. Preparation is outside the inference loop.
weights = pack_expert_weights(config, w13, w2, s13, s2, bias1, bias2)
out = dynamic_mxfp4_mega_moe(
    hidden_states, weights, global_expert_ids, topk_weights,
    workspace=workspace,
)
```

`pack_expert_weights` pads logical hidden dimensions and uses the same native
weight/bias codec as local MoE. The lower-level `ExpertWeights.pack` also accepts
already-padded raw matrices.

The lower-level API accepts quantized input views:

```python
views = workspace.input_views(num_tokens)
workspace.quantize(hidden_states)
views.expert_ids.copy_(global_expert_ids)
views.expert_weights.copy_(topk_weights)
out = workspace.run(weights, num_tokens)
```

Packed input rows contain `D/2` act bytes followed by `D/32` E8M0 scales;
scale storage is rounded to 16 bytes. D=2880 uses 1536 bytes per row, and D=7168
uses 3808. Quantization refreshes the padding bytes, including zero-input rows.
`MegaMoeInputViews` can also supply external contiguous routing tensors.

Allocate and warm up before graph capture. Stage1 factories reuse the selected tile/producer configuration
across token counts in the same bucket. The workspace supports sequential
invocations, different token-count graphs, and graph replay without host-side
epoch updates. Keep it alive until its kernels and graphs finish. Concurrent
invocations require separate workspaces. By default the returned output aliases
the workspace; supply `out=` to retain outputs separately. An output view may
have either logical-hidden or compute-hidden row stride.

## Implementation

The dependency direction is `mega_moe -> local_moe -> avelang`. Local MoE
owns the complete single-GPU implementation, including its W13/W2 tile
factories, activation, quantization, packing, scheduling and solution-ID fields.
It can import and run independently of MegaMoE and the symmetric heap.
MegaMoE reuses those local compute operations and adds transport, distributed
workspace management, persistent scheduling and cross-GPU result publication.

| Component | AveLang |
| --- | --- |
| ID / selection | `mega_moe/solutionid.py`, `config.py`, `dispatch.py` |
| Peer memory | `mega_moe/symmetric_heap.py` |
| Workspace | `mega_moe/workspace.py` |
| Input quantization | `mega_moe/quantization.py` |
| Count / plan / push | `mega_moe/token_shuffle_direct_push.py` |
| Persistent tickets | `mega_moe/scheduler.py` |
| Packed input / scale transpose | `mega_moe/input_mxfp4_packed.py` |
| Stage1 GEMM + activation | `local_moe/stage1.py`, `local_moe/weight_mxfp4.py`, `local_moe/activation.py` |
| Stage2 GEMM + BF16 C-shuffle | `local_moe/stage2.py` |
| Intermediate quantization / store / read | `local_moe/intermediate_mxfp4.py` |
| Stage execution | `mega_moe/stage1.py`, `stage2.py`; EP1 fusion in `api.py` |
| Result publication / sum | `mega_moe/route_output.py` |

For EP2/4/8, the first kernel uses 256 CTAs with fixed initial roles. CTA0
publishes counts and the destination plan; selected producers push activations and
per-M32 arrival masks. All CTAs then consume Stage1 tickets from eight independent
work heads. A CTA waits only for the rows it consumes. The 56-producer geometry
splits source/expert fragments with at least 64 rows across destination-local
producers. Other geometries retain one task per producer.

Stage1 uses M32 or M64 with 256 columns per projection. Its four/eight-wave
choice and the producer-count thresholds follow Petit. Scale rows are loaded and
repacked once per work item, then reused over the K loop. Act copies use
`raw_buffer_load_x4_lds`. Stage1 publishes its completed quantized tile using
scoped stores and an arrival bit. W2 launches 1280 CTAs and strides over further
tickets, writing uniquely owned BF16 route rows back to the source GPU.

The 128-CTA, eight-wave combine kernel joins system-scope output publication,
exchanges epochs, then sums top-k contributions in FP32 and rounds once to BF16.
The route buffer is overwritten on each invocation and does not need clearing.
Epoch admission and parity-buffered plans prevent reuse while a peer still uses
an earlier invocation.

EP1 fuses dispatch, both GEMMs and reduction into one 256-CTA kernel. It assigns
all Stage1 tickets before Stage2 tickets, so waiting for intermediate data cannot
prevent its producers from running. It shares the count/planner code with the
multi-rank path and performs local copies; Petit EP1 uses its separate pull
shuffle. The compute ticket order and EP1 reduction mapping follow Petit.

### DSL support and shared code

The port adds scoped u32 buffer atomics, compiler barriers, polling backoff and
GPU fences; permits cache-policy bits on LDS DMA and byte stores; and exposes
the return value of global/LDS `atomic_add`. Buffer offsets are lowered to their
32-bit hardware representation. Large integer literals retain 64-bit values
before explicit conversions, so >2 GiB workspace sizes remain valid.

JIT tensor shapes evaluated in factory annotations are materialized and included
in cache keys. Numeric captures are bound per helper before address-space
specialization. This prevents a caller's same-named constant from changing a
helper's dimensions or turning its local variables into immutable globals.

MegaMoE imports `make_stage1_compute` directly from `local_moe/stage1.py`, and
the K256 compute factory from `local_moe/stage2.py`. Shared Stage1 compute and
access helpers consume the selected `MegaMoeConfig` directly. `get_2stage_cfgs`
in `dispatch.py` selects the producer geometry, Stage1 tile M and wave count
together. Each workspace invocation calls it with the current token count;
wave M and LDS sizes derive from the selected configuration in `config.py`. Raw token counts are
not stored in the configuration, so equivalent launch buckets reuse kernels.
Quantization, intermediate storage, expert-weight packing/access and solution-ID
definitions also belong to `local_moe`. Local stage wrappers use these same
implementations independently of MegaMoE. Distributed readiness, task scheduling
and output publication remain in MegaMoE's own stage wrappers.

## Validation and benchmark

```bash
pytest -q test/examples/gemm/amdgpu/test_dynamic_mxfp4_mega_moe.py
python -m torch.distributed.run --standalone --nproc-per-node=8 \
  benchmark/moe/bench_megamoe.py --models gptoss120b dsv32 dsv4 \
  --tokens 8 128 1024 --scopes compute dynamic
```

The benchmark creates synthetic inputs, weights and fixed top-k routing.
`compute` starts from quantized input; `dynamic` also includes packed input
quantization. Each timing sample uses the maximum GPU event latency across
ranks, and the report gives the median across samples. Reports go to rank-zero
stdout or an explicit `--output` path.

Petit cross-checks, reference-library builds, scale-tail overlays and historical
reports live in `~/compare/avelang-petit/`. The repository tests are self-contained.
