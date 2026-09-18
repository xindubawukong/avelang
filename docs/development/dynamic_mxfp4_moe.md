# Dynamic MXFP4 Local MoE with Kimi support

History step 41: Kimi uses SiTU, sorted intermediate activations and a
weight-A K128 Stage2. The existing GPT-OSS/DSv3/DSv4 paths remain available.
Subsequent commits add split-K, SiTU scheduling, M64/grouped work, resident
input, weight prefetch, epilogue overlap, cache tuning and route reduction.
The precise mechanism and source references are in each commit message.

All four model profiles use the same seeded inputs, CPU/known-projection
references and graph checks. The final frozen Kimi tree is38bf8bf.
