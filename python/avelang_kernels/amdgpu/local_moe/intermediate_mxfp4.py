"""MXFP4 intermediate storage: allocation layout, Stage1 stores and Stage2 reads.

Activation and scale row indices are independent. Ordinary local MoE stores
activations in token/slot order and scales in sorted-route order; Kimi uses sorted
rows for both, while MegaMoE uses expert-pool rows. Callers select those rows,
check validity and publish readiness. The access helpers do not add barriers.
"""

from dataclasses import dataclass
from functools import cache

import avelang
import avelang.language as al
import torch

from .quantization import quantize_mxfp4_activation
from .scale_layout import ceildiv, scale_byte_offset, scale_byte_shape


@dataclass(frozen=True)
class IntermediateLayout:
    """Native scale tiles and either sorted or token/slot-major act."""

    route_capacity: int
    intermediate: int
    sorted_act: bool = False

    @property
    def scale_shape(self) -> tuple[int, int]:
        return (ceildiv(self.route_capacity, 256) * 256, ceildiv(self.intermediate, 256) * 8)

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
        if self.sorted_act:
            act = flat[: self.scale_offset].view(self.route_capacity, self.intermediate // 2)
        else:
            act = flat[: tokens * topk * self.intermediate // 2].view(tokens, topk, self.intermediate // 2)
        scales = flat[self.scale_offset : self.nbytes].view(scale_byte_shape(self.scale_shape[0], self.intermediate))
        return (act, scales)


@cache
def make_intermediate_store(intermediate, scale_columns, *, act_aux, scale_aux):
    I, SCALE_COLS = (intermediate, scale_columns)

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
        partner = al.amdgpu.get_dpp(packed, packed, 177, 15, 15, 0)
        if col_lane % 2 == 0:
            offset = act_base + act_row * (I // 2) + column_base // 2 + col_lane * 2
            al.amdgpu.raw_buffer_store_x1(packed | partner << 16, act_resource, offset, 0, act_aux)
        if col_lane % 8 == 0:
            col = column_base // 32 + col_lane // 8
            offset = scale_byte_offset(scale_row, col, al.convert(SCALE_COLS, al.u32))
            al.amdgpu.raw_buffer_store_u8(al.convert(exponent, al.u8), act_resource, offset, scale_base, scale_aux)

    return store_intermediate


@cache
def make_stage2_input_k256(words, act_aux):
    WORDS = words

    @avelang.jit
    def prefetch_stage2_input(
        act_resource: al.Tensor((4,), al.u32),
        act_offset: al.u32,
        act_base: al.u32,
        scale_offset: al.u32,
        scale_base: al.u32,
        valid: al.u1,
        k: al.u32,
    ) -> (al.Tensor((4,), al.u32), al.u32):
        offset = al.select(valid, act_offset + k * 128, al.convert(4294967280, al.u32))
        act = al.amdgpu.raw_buffer_load_x4(act_resource, offset, act_base, act_aux)
        scale = al.amdgpu.raw_buffer_load_x1(act_resource, scale_offset + k * 256, scale_base, act_aux)
        return (act, scale)

    @avelang.jit
    def read_stage2_input(
        storage: al.Tensor((WORDS,), al.u32), fragments: al.Tensor((2, 2, 4), al.u32), k: al.u32, lane: al.u32
    ):
        lds = al.view(storage, al.u32, al.make_layout((2, 32, 16, 4), (2048, 64, 4, 1)))
        for m in al.static_range(2):
            for half_k in al.static_range(2):
                row = m * 16 + lane % 16
                fragments[m, half_k] = lds[k % 2, row, lane // 16 + half_k * 4 ^ row & 15]

    return (prefetch_stage2_input, read_stage2_input)


@cache
def make_stage2_input_k128(intermediate, tile_m, scale_columns):
    I, BM, SC = (intermediate, tile_m, scale_columns)
    KT, MR, SX = (I // 128, BM // 16, BM // 32)
    DMA_WARPS, WORDS = (BM // 16, BM * 128)

    @avelang.jit
    def prefetch_resident_input(
        act_resource: al.Tensor((4,), al.u32),
        storage: al.Tensor((WORDS,), al.u32),
        block: al.u32,
        wave: al.u32,
        lane: al.u32,
    ):
        for k in al.static_range(KT):
            if wave < DMA_WARPS:
                row = wave * 16 + lane // 4
                source_vector = lane % 4 ^ row >> 1 & 3
                offset = (block * BM + row) * (I // 2) + k * 64 + source_vector * 16
                destination = (k * BM * 16 + wave * 256) * 4
                al.amdgpu.raw_buffer_load_x4_lds(act_resource, storage, 16, offset, 0, destination, 0)

    @avelang.jit
    def read_resident_input(
        storage: al.Tensor((WORDS,), al.u32), fragments: al.Tensor((MR, 4), al.u32), k: al.u32, lane: al.u32
    ):
        lds = al.view(storage, al.u32, al.make_layout((KT, BM, 4, 4), (BM * 16, 16, 4, 1)))
        for m in al.static_range(MR):
            row = m * 16 + lane % 16
            fragments[m] = lds[k, row, lane // 16 ^ row >> 1 & 3]

    @avelang.jit
    def load_input_scales(
        act_resource: al.Tensor((4,), al.u32),
        cached: al.Tensor((2,), al.u32),
        block: al.u32,
        scale_base: al.u32,
        k: al.u32,
        lane: al.u32,
    ):
        for m32 in al.static_range(SX):
            offset = (block * BM + m32 * 32) * SC + k // 2 * 256 + lane * 4
            cached[m32] = al.amdgpu.raw_buffer_load_x1(act_resource, offset, scale_base, 0)

    return (prefetch_resident_input, read_resident_input, load_input_scales)
