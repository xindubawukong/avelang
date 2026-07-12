from __future__ import annotations

from .config import (
    GemmConfig,
    LOAD_MODE_BASE_OFFSET,
    LOAD_MODE_DEFAULT,
    STAGGER_BY_M,
    STAGGER_BY_N,
    WGM_ROW_MAJOR,
    WGM_XCC,
    WGM_XCC_MAPPING8,
    WGM_XCC_MAPPING32,
)


CONFIGS: tuple[GemmConfig, ...] = (
    GemmConfig(
        name="m64n64k64_batch2_stagger_sched1",
        group_m=64,
        group_n=64,
        group_k=64,
        partition_m=2,
        partition_n=2,
        partition_k=1,
        num_batch_k=2,
        read_rows_a=2,
        read_rows_b=2,
        pad_a_bytes=32,
        pad_b_bytes=32,
        wgm_mode=WGM_XCC,
        stagger_mask=31,
        stagger_stride=2,
        stagger_by=STAGGER_BY_N,
        loop_scheduler=1,
        load_mode=LOAD_MODE_DEFAULT,
        store_vec=1,
        prefetch_before_zero=False,
        prefetch1_before_read=False,
        pipeline_interleave=True,
        source="1024 tuned batch2 stagger variant",
    ),
    GemmConfig(
        name="m128n128k64_batch2_stagger_sched0",
        group_m=128,
        group_n=128,
        group_k=64,
        partition_m=2,
        partition_n=2,
        partition_k=1,
        num_batch_k=2,
        read_rows_a=4,
        read_rows_b=4,
        pad_a_bytes=32,
        pad_b_bytes=32,
        wgm_mode=WGM_XCC,
        stagger_mask=31,
        stagger_stride=2,
        stagger_by=STAGGER_BY_N,
        loop_scheduler=0,
        load_mode=LOAD_MODE_BASE_OFFSET,
        store_vec=2,
        prefetch_before_zero=True,
        prefetch1_before_read=True,
        pipeline_interleave=False,
        source="2048 tuned batch2 stagger variant",
    ),
    GemmConfig(
        name="m128n128k64_batch2_mapping32_sched0",
        group_m=128,
        group_n=128,
        group_k=64,
        partition_m=2,
        partition_n=2,
        partition_k=1,
        num_batch_k=2,
        read_rows_a=4,
        read_rows_b=4,
        pad_a_bytes=32,
        pad_b_bytes=32,
        wgm_mode=WGM_XCC_MAPPING32,
        stagger_mask=0,
        stagger_stride=0,
        stagger_by=STAGGER_BY_N,
        loop_scheduler=0,
        load_mode=LOAD_MODE_DEFAULT,
        store_vec=2,
        prefetch_before_zero=False,
        prefetch1_before_read=False,
        pipeline_interleave=False,
        source="generic batch2 mapping32 variant",
    ),
    GemmConfig(
        name="m224n256k64_batch4_rowmajor_x4",
        group_m=224,
        group_n=256,
        group_k=64,
        partition_m=2,
        partition_n=2,
        partition_k=1,
        num_batch_k=4,
        read_rows_a=1,
        read_rows_b=8,
        pad_a_bytes=8,
        pad_b_bytes=8,
        wgm_mode=WGM_ROW_MAJOR,
        stagger_mask=0,
        stagger_stride=0,
        stagger_by=STAGGER_BY_M,
        loop_scheduler=1,
        load_mode=LOAD_MODE_BASE_OFFSET,
        store_vec=4,
        prefetch_before_zero=True,
        prefetch1_before_read=False,
        pipeline_interleave=True,
        source="4096 tuned batch4 row-major variant",
    ),
    GemmConfig(
        name="m224n256k64_batch4_mapping8_x4",
        group_m=224,
        group_n=256,
        group_k=64,
        partition_m=2,
        partition_n=2,
        partition_k=1,
        num_batch_k=4,
        read_rows_a=1,
        read_rows_b=8,
        pad_a_bytes=8,
        pad_b_bytes=8,
        wgm_mode=WGM_XCC_MAPPING8,
        stagger_mask=0,
        stagger_stride=0,
        stagger_by=STAGGER_BY_M,
        loop_scheduler=1,
        load_mode=LOAD_MODE_BASE_OFFSET,
        store_vec=4,
        prefetch_before_zero=True,
        prefetch1_before_read=False,
        pipeline_interleave=True,
        source="8192 tuned batch4 mapping8 x4 variant",
    ),
    GemmConfig(
        name="m224n256k64_batch4_mapping8_x2",
        group_m=224,
        group_n=256,
        group_k=64,
        partition_m=2,
        partition_n=2,
        partition_k=1,
        num_batch_k=4,
        read_rows_a=1,
        read_rows_b=8,
        pad_a_bytes=8,
        pad_b_bytes=8,
        wgm_mode=WGM_XCC_MAPPING8,
        stagger_mask=0,
        stagger_stride=0,
        stagger_by=STAGGER_BY_M,
        loop_scheduler=1,
        load_mode=LOAD_MODE_DEFAULT,
        store_vec=2,
        prefetch_before_zero=False,
        prefetch1_before_read=False,
        pipeline_interleave=False,
        source="16384 tuned batch4 mapping8 x2 variant",
    ),
)

CONFIG_BY_NAME = {config.name: config for config in CONFIGS}
CONFIG_BY_KEY = {config.key: config for config in CONFIGS}

DEFAULT_CONFIG_BY_SHAPE = {
    (1024, 1024, 1024): "m64n64k64_batch2_stagger_sched1",
    (2048, 2048, 2048): "m128n128k64_batch2_stagger_sched0",
    (4096, 4096, 4096): "m224n256k64_batch4_rowmajor_x4",
    (8192, 8192, 8192): "m224n256k64_batch4_mapping8_x4",
    (16384, 16384, 16384): "m224n256k64_batch4_mapping8_x2",
}


def get_config(name: str) -> GemmConfig:
    try:
        return CONFIG_BY_NAME[name]
    except KeyError as exc:
        choices = ", ".join(sorted(CONFIG_BY_NAME))
        raise ValueError(f"Unknown GEMM config {name!r}. Available: {choices}") from exc


def enumerate_configs(m: int, n: int, k: int) -> list[GemmConfig]:
    return [config for config in CONFIGS if config.supports(m, n, k)]


def default_config(m: int, n: int, k: int) -> GemmConfig:
    name = DEFAULT_CONFIG_BY_SHAPE.get((m, n, k))
    if name is not None:
        return get_config(name)

    candidates = enumerate_configs(m, n, k)
    if not candidates:
        raise ValueError(f"No AMDGPU GEMM config supports M={m}, N={n}, K={k}.")
    return candidates[0]
