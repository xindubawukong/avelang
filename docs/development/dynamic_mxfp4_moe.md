# Dynamic MXFP4 local MoE

BF16 input/output, dynamic MXFP4 activations and native MXFP4 weights on gfx950.
Supports GPT-OSS, DeepSeek V3/V4 and Kimi K3 latent MoE from the first revision.
The public API is `get_2stage_cfgs`, `ExpertWeights.pack`, `Routing`,
`MoeWorkspace.allocate` and `dynamic_mxfp4_moe`. Local MoE is independent of Mega.

This revision uses weight-as-A for K256 and K128. Stage1 produces a quantized
intermediate; Stage2 applies the route weights and writes BF16 outputs. Initial
Kimi support gathers token/slot activations explicitly; later revisions store
Kimi activations in sorted order. Scale layout and pipeline changes arrive in
their own commits.

Implemented optimization steps in this revision:

- k256 lds
- k256 prefetch
- native scales
- sorted act
- stage1 dma
- stage1 pipeline
- w13 resources


Validation in this reconstructed history is offline gfx950 compilation only.
No GPU execution or performance measurements were performed. See
`~/compare/moe-history-rebuild-20260917` for per-commit compile evidence.
