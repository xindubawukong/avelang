# Local MoE after the dev-megamoe milestone

The exact dev-megamoe-aligned tree is the preceding milestone d57fbc4.
This section adds later shared mechanisms before Kimi-only functionality.
Stage1 and Stage2 still support GPT-OSS, DSv3 and DSv4.

## Added mechanisms

- [moe][rocm] Use native MXFP4 intermediate scale layout
- [moe][rocm] Support independent stage1 and stage2 weight policies

Each commit has numerical, graph-replay and four-scope timing evidence under
~/compare/local-moe-two-milestones-20260918.
