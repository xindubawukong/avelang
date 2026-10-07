"""Native gfx950 packed FP4 conversion."""

import avelang
import avelang.language as al
import pytest
import torch
from avelang.testing import has_gfx950


pytestmark = pytest.mark.skipif(
    not has_gfx950(), reason="Requires gfx950 FP4 conversion",
)


@avelang.jit
def kernel_pack_fp4(
    src: al.Tensor((64, 2), al.f32),
    old: al.Tensor((64,), al.u32),
    scale: al.Tensor((64,), al.f32),
    out: al.Tensor((64,), al.u32),
    byte_sel: al.constexpr,
):
    lane = al.thread_id(0)
    out[lane] = al.amdgpu.cvt_scalef32_pk_fp4_f32(
        old[lane], src[lane, 0], src[lane, 1], scale[lane], byte_sel,
    )


def test_fp4_pack():
    fp4_values = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6])
    codes = torch.arange(128).reshape(64, 2) % 16
    scales = 2.0 ** ((torch.arange(64) % 7) - 3)
    src = (fp4_values[codes] * scales[:, None]).cuda()
    scales = scales.cuda()
    old = torch.full((64,), 0xA1B2C3D4, dtype=torch.uint32, device="cuda")
    out = torch.empty_like(old)
    packed = codes[:, 0] | (codes[:, 1] << 4)
    for byte_sel in range(4):
        kernel_pack_fp4[lambda: ((1, 1, 1), (64, 1, 1))](src, old, scales, out, byte_sel)
        expected = (0xA1B2C3D4 & ~(255 << (8 * byte_sel))) | (packed << (8 * byte_sel))
        torch.testing.assert_close(out.cpu().to(torch.int64), expected, rtol=0, atol=0)
