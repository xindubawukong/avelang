"""Native scaled FP4 MFMA instruction contracts used by dynamic MoE."""

import avelang
import avelang.language as al
import pytest
import torch
from avelang.testing import has_gfx950


pytestmark = pytest.mark.skipif(
    not has_gfx950(), reason="Requires gfx950 scaled MFMA",
)


@avelang.jit
def kernel_scaled_mfma(
    a: al.Tensor((64, 4), al.u32),
    sa: al.Tensor((64,), al.u32),
    b: al.Tensor((64, 4), al.u32),
    sb: al.Tensor((64,), al.u32),
    c: al.Tensor((64, 4), al.f32),
    out: al.Tensor((5, 64, 4), al.f32),
):
    lane = al.thread_id(0)
    out[0, lane] = al.amdgpu.mfma_scale_16x16x128_fp4(
        a[lane], sa[lane], b[lane], sb[lane], c[lane], 1, 2,
    )
    for selector in al.range(4):
        out[selector + 1, lane] = al.amdgpu.mfma_scale_16x16x128_fp4(
            a[lane], sa[lane], b[lane], sb[lane], c[lane], selector, 3 - selector,
        )


def pack_fragments(codes):
    # Per-lane source: row = lane % 16, K32 group = lane // 16.
    words = sum(codes.reshape(16, 4, 4, 8)[..., i] << (4 * i) for i in range(8))
    return words.permute(1, 0, 2).reshape(64, 4).to(torch.uint32).cuda()


def test_scaled_mfma():
    fp4_values = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6])
    generator = torch.Generator().manual_seed(123)
    a = torch.randint(0, 16, (16, 128), generator=generator)
    b = torch.randint(0, 16, (16, 128), generator=generator)
    c = torch.arange(256, dtype=torch.float32).reshape(16, 16) / 4
    c_fragments = c.reshape(16, 4, 4).permute(1, 0, 2).reshape(64, 4).contiguous().cuda()
    sa = torch.full((64,), 0x807F7E7D, dtype=torch.uint32, device="cuda")
    sb = torch.full((64,), 0x7D7E7F80, dtype=torch.uint32, device="cuda")
    out = torch.empty((5, 64, 4), device="cuda")
    kernel_scaled_mfma[lambda: ((1, 1, 1), (64, 1, 1))](
        pack_fragments(a), sa, pack_fragments(b), sb, c_fragments, out,
    )
    actual = out.cpu().reshape(5, 4, 16, 4).permute(0, 2, 1, 3).reshape(5, 16, 16)
    # E8M0 exponent bias is 127; fixed (1, 2), then (s, 3 - s).
    scales = torch.tensor([1 / 4, 1 / 16, 1 / 4, 1, 4]).view(5, 1, 1)
    # lane % 16 selects the output's B row, so the reference is B @ A.T.
    expected = (fp4_values[b] @ fp4_values[a].T) * scales + c
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
