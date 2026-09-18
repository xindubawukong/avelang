"""MXFP4 intermediate storage: allocation layout, Stage1 stores and Stage2 reads.

Activations use token/slot order; scales use sorted-route order. Callers
select the row indices and check validity. The access helpers do not add barriers.
"""

from dataclasses import dataclass
from functools import cache

import avelang
import avelang.language as al
import torch

from .quantization import quantize_mxfp4_activation
from .scale_layout import ceildiv


@dataclass(frozen=True)
class IntermediateLayout:
    """Row-major sorted-route scales and token/slot-major act."""

    route_capacity: int
    intermediate: int

    @property
    def scale_shape(self) -> tuple[int, int]:
        return ceildiv(self.route_capacity, 256) * 256, ceildiv(self.intermediate, 256) * 8

    @property
    def scale_offset(self) -> int:
        return self.route_capacity * self.intermediate // 2

    @property
    def nbytes(self) -> int:
        rows, cols = self.scale_shape
        return self.scale_offset + rows * cols

    def views(self, buffer: torch.Tensor, tokens: int, topk: int):
        if buffer.dtype != torch.uint8 or not buffer.is_contiguous() or buffer.numel() < self.nbytes:
            raise ValueError("workspace must be a contiguous uint8 tensor with sufficient capacity")
        if tokens * topk > self.route_capacity:
            raise ValueError("workspace route capacity is smaller than tokens * topk")
        flat = buffer.view(-1)
        act = flat[: tokens * topk * self.intermediate // 2].view(tokens, topk, self.intermediate // 2)
        scales = flat[self.scale_offset : self.nbytes].view(self.scale_shape)
        return act, scales


@cache
def make_intermediate_store(intermediate, scale_columns, *, act_aux, scale_aux):
    I, SCALE_COLS = intermediate, scale_columns

    @avelang.jit
    def store_intermediate(
        act: al.Tensor((4,), al.f32),
        act_resource: al.Tensor((4,), al.u32),
        act_row: al.u32,
        scale_row: al.u32,
        column_base: al.u32,
        col_lane: al.u32,
        act_base: al.u32,
        scale_base: al.u32,
    ):
        packed, exponent = quantize_mxfp4_activation(act)
        partner = al.amdgpu.get_dpp(packed, packed, 0xB1, 15, 15, 0)
        if col_lane % 2 == 0:
            offset = act_base + act_row * (I // 2) + column_base // 2 + col_lane * 2
            al.amdgpu.raw_buffer_store_x1(packed | (partner << 16), act_resource, offset, 0, act_aux)
        # Four adjacent K32 scales share one aligned row-major u32 store.
        scale0 = al.shuffle(exponent, 0, 32)
        scale1 = al.shuffle(exponent, 8, 32)
        scale2 = al.shuffle(exponent, 16, 32)
        scale3 = al.shuffle(exponent, 24, 32)
        if col_lane % 32 == 0:
            scales = scale0 | (scale1 << 8) | (scale2 << 16) | (scale3 << 24)
            offset = scale_row * SCALE_COLS + column_base // 32 + col_lane // 8
            al.amdgpu.raw_buffer_store_x1(scales, act_resource, offset, scale_base, scale_aux)

    return store_intermediate


@cache
def make_stage2_input_k256(intermediate, words):
    WORDS, SCALE_COLS = words, intermediate // 32

    @avelang.jit
    def prefetch_stage2_input(
        act_resource: al.Tensor((4,), al.u32),
        act_offset: al.u32,
        act_base: al.u32,
        scale_offset: al.u32,
        scale_base: al.u32,
        valid: al.u1,
        k: al.u32,
        lane: al.u32,
    ) -> (al.Tensor((4,), al.u32), al.u32):
        offset = al.select(valid, act_offset + k * 128, al.convert(0xFFFFFFF0, al.u32))
        act = al.amdgpu.raw_buffer_load_x4(act_resource, offset, act_base, 0)
        # One aligned u32 load per lane, then redistribute its bytes to MFMA.
        load_row, load_col = lane % 32, k * 8 + (lane // 32) * 4
        loaded = al.amdgpu.raw_buffer_load_x1(
            act_resource, scale_offset + load_row * SCALE_COLS + load_col, scale_base, 0
        )
        row, shift = lane % 16, (lane // 16) * 8
        scale = (
            ((al.shuffle(loaded, row, 64) >> shift) & 255)
            | (((al.shuffle(loaded, row + 16, 64) >> shift) & 255) << 8)
            | (((al.shuffle(loaded, row + 32, 64) >> shift) & 255) << 16)
            | (((al.shuffle(loaded, row + 48, 64) >> shift) & 255) << 24)
        )
        return act, scale

    @avelang.jit
    def read_stage2_input(
        storage: al.Tensor((WORDS,), al.u32),
        fragments: al.Tensor((2, 2, 4), al.u32),
        k: al.u32,
        lane: al.u32,
    ):
        lds = al.view(storage, al.u32, al.make_layout((2, 32, 16, 4), (2048, 64, 4, 1)))
        for m in al.static_range(2):
            for half_k in al.static_range(2):
                row = m * 16 + lane % 16
                fragments[m, half_k] = lds[k % 2, row, (lane // 16 + half_k * 4) ^ (row & 15)]

    return prefetch_stage2_input, read_stage2_input
