# Local MoE after the dev-megamoe milestone

The exact dev-megamoe-aligned tree is the preceding milestone d57fbc4.
This section adds later shared mechanisms before Kimi-only functionality.
Stage1 and Stage2 still support GPT-OSS, DSv3 and DSv4.

## Added mechanisms

- [moe][rocm] Use native MXFP4 intermediate scale layout
- [moe][rocm] Support independent stage1 and stage2 weight policies
- [moe][rocm] Prefetch stage2 weights before input LDS stores
- [moe][rocm] Remove redundant stage2 K-loop barriers
- [moe][rocm] Interleave stage1 W13 prefetch in four phases
- [moe][rocm] Use partial VMEM waits in the stage1 pipeline
- [moe][rocm] Prefetch stage2 epilogue inputs into registers
- [moe][rocm] Use packed BF16 stores in the stage2 epilogue
- [moe][rocm] Use wave-uniform stage2 output writeback
- [moe][rocm] Use an XOR LDS layout for stage2 C-shuffle

Each commit has numerical, graph-replay and four-scope timing evidence under
~/compare/local-moe-two-milestones-20260918.
