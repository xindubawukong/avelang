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
from .scale_layout import ceildiv, load_scale_byte


@dataclass(frozen=True)
class IntermediateLayout:
    """Row-major scales and token/slot-major act."""

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
def make_intermediate_store():
    @avelang.jit
    def store_intermediate(
        act: al.Tensor((4,), al.f32),
        resource: al.Tensor((4,), al.u32),
        act_row: al.u32,
        scale_row: al.u32,
        column_base: al.u32,
        col_lane: al.u32,
        scale_base: al.u32,
        intermediate: al.u32,
    ):
        packed, exponent = quantize_mxfp4_activation(act)
        partner = al.amdgpu.get_dpp(packed, packed, 0xB1, 15, 15, 0)
        if col_lane % 2 == 0:
            offset = act_row * (intermediate // 2) + column_base // 2 + col_lane * 2
            al.amdgpu.raw_buffer_store_x1(packed | (partner << 16), resource, offset, 0, 0)
        if col_lane % 8 == 0:
            offset = scale_row * (intermediate // 32) + column_base // 32 + col_lane // 8
            al.amdgpu.raw_buffer_store_u8(al.convert(exponent, al.u8), resource, offset, scale_base, 0)

    return store_intermediate


@cache
def make_stage2_input(config):
    TOPK = config.topk

    @avelang.jit
    def load_intermediate(
        act: al.Tensor((4,), al.u32),
        routes: al.Tensor((4,), al.u32),
        intermediate: al.u32,
        tokens: al.u32,
        block: al.u32,
        m16: al.u32,
        k128: al.u32,
        lane: al.u32,
        scale_base: al.u32,
    ) -> (al.Tensor((4,), al.u32), al.u32):
        row = m16 * 16 + lane % 16
        route = al.amdgpu.raw_buffer_load_x1(routes, row * 4, 0, 0)
        token, slot = route & 0xFFFFFF, route >> 24
        values = al.full((4,), 0, al.u32)
        scale = al.convert(0, al.u32)
        if token < tokens and slot < TOPK:
            offset = (token * TOPK + slot) * (intermediate // 2) + k128 * 64 + lane // 16 * 16
            values = al.amdgpu.raw_buffer_load_x4(act, offset, 0, 0)
            scale = load_scale_byte(act, scale_base + (block * 32 + row) * (intermediate // 32) + k128 * 4 + lane // 16)
        return values, scale

    return load_intermediate
