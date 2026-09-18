"""MoE activation behavior around the OpenAI clipping boundaries."""

import avelang
import avelang.language as al
import pytest
import torch
from avelang.testing import has_rocm
from avelang_kernels.amdgpu.local_moe.activation import openai_swiglu, silu_dot


@avelang.jit
def activate_pairs(pairs: al.Tensor((64, 2), al.f32), silu: al.Tensor((64,), al.f32), swiglu: al.Tensor((64,), al.f32)):
    lane = al.thread_id(0)
    gate, up = pairs[lane, 0], pairs[lane, 1]
    silu[lane] = silu_dot(gate, up)
    swiglu[lane] = openai_swiglu(gate, up)


@pytest.mark.skipif(not has_rocm(), reason="MoE activations use AMD rcp")
def test_activation_clipping_and_silu_dot():
    edges = torch.tensor([-20.0, -7.01, -7.0, -6.99, 0.0, 6.99, 7.0, 7.01])
    pairs = torch.cartesian_prod(edges, edges).cuda()
    silu = torch.empty(64, device="cuda")
    swiglu = torch.empty_like(silu)
    activate_pairs[lambda: ((1, 1, 1), (64, 1, 1))](pairs, silu, swiglu)
    gate, up = pairs[:, 0], pairs[:, 1]
    expected_silu = torch.nn.functional.silu(gate) * up
    clipped_gate = gate.clamp_max(7)
    expected_swiglu = clipped_gate * torch.sigmoid(1.702 * clipped_gate) * (up.clamp(-7, 7) + 1)
    torch.testing.assert_close(silu, expected_silu, rtol=2e-6, atol=1e-6)
    torch.testing.assert_close(swiglu, expected_swiglu, rtol=2e-6, atol=1e-6)
