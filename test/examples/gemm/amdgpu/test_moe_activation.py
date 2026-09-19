"""MoE activation behavior around the OpenAI clipping boundaries."""

import avelang
import avelang.language as al
import pytest
import torch
from avelang.testing import has_rocm
from avelang_kernels.amdgpu.local_moe.activation import openai_swiglu, silu_dot, situ_v2


@avelang.jit
def activate_pairs(pairs: al.Tensor((64, 2), al.f32), silu: al.Tensor((64,), al.f32), swiglu: al.Tensor((64,), al.f32)):
    lane = al.thread_id(0)
    gate, up = (pairs[lane, 0], pairs[lane, 1])
    silu[lane] = silu_dot(gate, up)
    swiglu[lane] = openai_swiglu(gate, up)


@avelang.jit
def activate_situ(pairs: al.Tensor((64, 2), al.f32), out: al.Tensor((64,), al.f32)):
    lane = al.thread_id(0)
    out[lane] = situ_v2(pairs[lane, 0], pairs[lane, 1])


@pytest.mark.skipif(not has_rocm(), reason="MoE activations use AMD rcp")
def test_situ_signs_zero_and_saturation():
    edges = torch.tensor([-1000.0, -100.0, -7.0, -0.0, 0.01, 7.0, 100.0, 1000.0])
    pairs = torch.cartesian_prod(edges, edges).cuda()
    out = torch.empty(64, device="cuda")
    activate_situ[lambda: ((1, 1, 1), (64, 1, 1))](pairs, out)
    gate, up = (pairs[:, 0].double(), pairs[:, 1].double())
    expected = 100 * torch.tanh(gate / 4) * torch.sigmoid(gate) * torch.tanh(up / 25)
    torch.testing.assert_close(out.double(), expected, rtol=2e-05, atol=2e-06)


@pytest.mark.skipif(not has_rocm(), reason="MoE activations use AMD rcp")
def test_activation_clipping_and_silu_dot():
    edges = torch.tensor([-20.0, -7.01, -7.0, -6.99, 0.0, 6.99, 7.0, 7.01])
    pairs = torch.cartesian_prod(edges, edges).cuda()
    silu = torch.empty(64, device="cuda")
    swiglu = torch.empty_like(silu)
    activate_pairs[lambda: ((1, 1, 1), (64, 1, 1))](pairs, silu, swiglu)
    gate, up = (pairs[:, 0], pairs[:, 1])
    expected_silu = torch.nn.functional.silu(gate) * up
    clipped_gate = gate.clamp_max(7)
    expected_swiglu = clipped_gate * torch.sigmoid(1.702 * clipped_gate) * (up.clamp(-7, 7) + 1)
    torch.testing.assert_close(silu, expected_silu, rtol=2e-06, atol=1e-06)
    torch.testing.assert_close(swiglu, expected_swiglu, rtol=2e-06, atol=1e-06)
