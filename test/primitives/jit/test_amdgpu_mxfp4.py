"""Native gfx950 MXFP4 instruction contracts used by dynamic MoE."""

from functools import cache

import avelang
import avelang.language as al
import pytest
import torch


def has_gfx950():
    return bool(torch.version.hip and torch.cuda.is_available()) and (
        torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName.split(":")[0] == "gfx950"
    )


pytestmark = pytest.mark.skipif(not has_gfx950(), reason="Native MXFP4 requires gfx950")


@cache
def make_pack_kernel(byte_sel):
    @avelang.jit
    def kernel(
        src: al.Tensor((64, 2), al.f32),
        old: al.Tensor((64,), al.u32),
        scales: al.Tensor((64,), al.f32),
        out: al.Tensor((64,), al.u32),
    ):
        lane = al.thread_id(0)
        out[lane] = al.amdgpu.cvt_scalef32_pk_fp4_f32(old[lane], src[lane, 0], src[lane, 1], scales[lane], byte_sel)

    return kernel


@cache
def make_mfma_kernel(sel_a, sel_b):
    @avelang.jit
    def kernel(
        a: al.Tensor((64, 4), al.u32),
        sa: al.Tensor((64,), al.u32),
        b: al.Tensor((64, 4), al.u32),
        sb: al.Tensor((64,), al.u32),
        c: al.Tensor((64, 4), al.f32),
        out: al.Tensor((64, 4), al.f32),
    ):
        lane = al.thread_id(0)
        out[lane] = al.amdgpu.mfma_scale_16x16x128_fp4(a[lane], sa[lane], b[lane], sb[lane], c[lane], sel_a, sel_b)

    return kernel


FP4_VALUES = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6])


@pytest.mark.parametrize("byte_sel", range(4))
def test_fp4_pack_preserves_other_bytes(byte_sel):
    codes = torch.arange(128).reshape(64, 2) % 16
    scales = 2.0 ** ((torch.arange(64) % 7) - 3)
    src = FP4_VALUES[codes] * scales[:, None]
    old = torch.full((64,), 0xA1B2C3D4, dtype=torch.uint32, device="cuda")
    out = torch.empty_like(old)
    make_pack_kernel(byte_sel)[lambda: ((1, 1, 1), (64, 1, 1))](src.cuda(), old, scales.cuda(), out)
    packed = codes[:, 0] | (codes[:, 1] << 4)
    expected = (0xA1B2C3D4 & ~(255 << (8 * byte_sel))) | (packed << (8 * byte_sel))
    torch.testing.assert_close(out.cpu().to(torch.int64), expected, rtol=0, atol=0)


def pack_fragments(codes):
    # Per-lane source: row = lane % 16, K32 group = lane // 16.
    words = sum(codes.reshape(16, 4, 4, 8)[..., i] << (4 * i) for i in range(8))
    return words.permute(1, 0, 2).reshape(64, 4).to(torch.uint32).cuda()


@pytest.mark.parametrize("sel_a,sel_b", [(0, 0), (1, 2), (2, 3), (3, 1)])
def test_scaled_mfma_fragment_mapping(sel_a, sel_b):
    generator = torch.Generator().manual_seed(123)
    a = torch.randint(0, 16, (16, 128), generator=generator)
    b = torch.randint(0, 16, (16, 128), generator=generator)
    c = torch.arange(256, dtype=torch.float32).reshape(16, 16) / 4
    c_fragments = c.reshape(16, 4, 4).permute(1, 0, 2).reshape(64, 4).contiguous().cuda()
    sa = torch.full((64,), 0x807F7E7D, dtype=torch.uint32, device="cuda")
    sb = torch.full((64,), 0x7D7E7F80, dtype=torch.uint32, device="cuda")
    out = torch.empty_like(c_fragments)
    make_mfma_kernel(sel_a, sel_b)[lambda: ((1, 1, 1), (64, 1, 1))](
        pack_fragments(a), sa, pack_fragments(b), sb, c_fragments, out
    )
    actual = out.cpu().reshape(4, 16, 4).permute(1, 0, 2).reshape(16, 16)
    scale = 2.0 ** ((125 + sel_a - 127) + (128 - sel_b - 127))
    # The source project uses weights as A, activations as B: lane % 16
    # selects the output's token row (B), not the weight row (A).
    expected = (FP4_VALUES[b] @ FP4_VALUES[a].T) * scale + c
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
