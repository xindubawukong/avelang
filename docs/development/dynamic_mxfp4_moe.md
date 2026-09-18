# Dynamic MXFP4 Local MoE: optimization history

Current step 12: [moe][rocm] Prefetch stage2 epilogue inputs.

This prefix targets Petit dev-megamoe before d97fb857. The public local ID
has one shared cache policy at bit40 and the Stage1 tile at bit41. Stage2 is
K256. The final dev-megamoe and Kimi trees are frozen separately in TARGETS.json
under ~/compare/local-moe-two-milestones-20260918.

Use get_2stage_cfgs, ExpertWeights.pack, Routing and MoeWorkspace to execute
dynamic_mxfp4_moe. Stage1 selects expert-routed inputs, computes gate/up and
quantizes intermediates; Stage2 applies W2 and routing weights. Input preparation
uses separate AITER quantization and scale sorting. This step supports GPT-OSS,
DSv3 and DSv4; Kimi support and post-Kimi shared optimizations arrive later.

The benchmark uses identical seeded raw inputs and repacks for each historical
layout. Stage2 includes output clearing and dynamic includes preparation.
