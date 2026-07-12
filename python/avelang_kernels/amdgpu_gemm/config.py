from __future__ import annotations

from dataclasses import dataclass


WGM_ROW_MAJOR = 0
WGM_XCC = 1
WGM_XCC_MAPPING8 = 2
WGM_XCC_MAPPING32 = 3

STAGGER_BY_M = 0
STAGGER_BY_N = 1

LOAD_MODE_DEFAULT = 0
LOAD_MODE_BASE_OFFSET = 1


@dataclass(frozen=True)
class GemmConfig:
    name: str
    group_m: int
    group_n: int
    group_k: int
    partition_m: int
    partition_n: int
    partition_k: int
    num_batch_k: int
    read_rows_a: int
    read_rows_b: int
    pad_a_bytes: int
    pad_b_bytes: int
    wgm_mode: int
    stagger_mask: int
    stagger_stride: int
    stagger_by: int
    loop_scheduler: int
    load_mode: int
    store_vec: int
    prefetch_before_zero: bool = False
    prefetch1_before_read: bool = False
    pipeline_interleave: bool = False
    source: str = ""

    @property
    def key(self) -> tuple[int, ...]:
        return (
            self.group_m,
            self.group_n,
            self.group_k,
            self.partition_m,
            self.partition_n,
            self.partition_k,
            self.num_batch_k,
            self.read_rows_a,
            self.read_rows_b,
            self.pad_a_bytes,
            self.pad_b_bytes,
            self.wgm_mode,
            self.stagger_mask,
            self.stagger_stride,
            self.stagger_by,
            self.loop_scheduler,
            self.load_mode,
            self.store_vec,
            int(self.prefetch_before_zero),
            int(self.prefetch1_before_read),
            int(self.pipeline_interleave),
        )

    def supports(self, m: int, n: int, k: int) -> bool:
        if k % self.group_k != 0:
            return False
        if self.num_batch_k == 2:
            return m % self.group_m == 0 and n % self.group_n == 0
        if self.num_batch_k == 4:
            # The 224x256 kernels use ceil-div M groups and buffer-size checks
            # on raw buffer accesses, but require full N tiles for vector stores.
            return n % self.group_n == 0
        return False
