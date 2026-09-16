import unittest

import avelang
import avelang.language as S
import pytest
import torch
from avelang.testing import has_rocm


@avelang.jit
def kernel_atomic_add_i32(value: S.i32, out: S.Tensor((1,), S.i32)):
    offset = S.convert(0, S.i32)
    S.amdgpu.atomic_add(offset, value, out, 1)


@avelang.jit
def kernel_atomic_add_packed_bf16(
    value: S.Tensor((2,), S.bf16),
    out: S.Tensor((2,), S.bf16),
):
    offset = S.convert(0, S.i32)
    packed_value = S.view(value, S.Tensor((1, 2), S.bf16))[0]
    S.amdgpu.atomic_add(offset, packed_value, out, 1)


@avelang.jit
def buffer_atomic_masked(out: S.Tensor((2,), S.bf16)):
    tid = S.thread_id(0)
    pair = S.full((1, 2), 1, S.bf16)
    resource = S.amdgpu.make_rsrc(out, 4)
    offset = S.select(tid < 32, S.convert(0, S.u32), S.convert(4, S.u32))
    S.amdgpu.raw_buffer_atomic_add_bf16x2(pair[0], resource, offset)


@avelang.jit
def fetch_add_shared_kernel(out: S.Tensor((64,), S.u32)):
    tid = S.convert(S.thread_id(0), S.u32)
    storage = S.make_shared((4,), S.u32)
    counter = S.subview(storage, (2,), (2,), (1,))
    if tid == 0:
        counter[0] = S.convert(0, S.u32)
    S.syncthreads()
    out[tid] = S.amdgpu.atomic_add(S.convert(0, S.u32), S.convert(1, S.u32), counter, 0)


@avelang.jit
def buffer_integer_atomics(
    values: S.Tensor((2,), S.u32),
    counters: S.Tensor((2,), S.u32),
    old_values: S.Tensor((128,), S.u32),
    aux: S.constexpr,
    scope: S.constexpr,
):
    tid = S.thread_id(0)
    resource = S.amdgpu.make_rsrc(counters, 8)
    S.amdgpu.compiler_barrier()
    S.amdgpu.fence(1, scope)
    old_values[tid] = S.amdgpu.raw_buffer_atomic_add_u32(values[0], resource, 0, 0, aux)
    old_values[64 + tid] = S.amdgpu.raw_buffer_atomic_or_u32(values[1], resource, 4, 0, aux)
    S.amdgpu.s_sleep(1)
    S.amdgpu.fence(0, scope)
    S.amdgpu.fence(2, scope)
    S.amdgpu.compiler_barrier()


@unittest.skipUnless(has_rocm(), "Requires ROCm/HIP with an AMD GPU.")
class TestAMDGPUAtomicAdd(unittest.TestCase):
    def test_i32_atomic_add(self):
        out = torch.tensor([11], dtype=torch.int32, device="cuda")

        kernel_atomic_add_i32[lambda: ((1, 1, 1), (1, 1, 1))](7, out)

        self.assertEqual(out.cpu().item(), 18)

    def test_packed_bf16_atomic_add(self):
        value = torch.tensor([1.5, -2.0], dtype=torch.bfloat16, device="cuda")
        out = torch.tensor([3.0, 4.0], dtype=torch.bfloat16, device="cuda")

        kernel_atomic_add_packed_bf16[lambda: ((1, 1, 1), (1, 1, 1))](value, out)

        expected = torch.tensor([4.5, 2.0], dtype=torch.bfloat16, device="cuda")
        self.assertTrue(torch.equal(out, expected), f"Expected {expected}, got {out}")

    @unittest.skipUnless(
        has_rocm()
        and torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName.split(":")[0] == "gfx950",
        "Packed BF16 buffer atomics require gfx950",
    )
    def test_buffer_atomic_masks_invalid_routes(self):
        out = torch.zeros(2, device="cuda", dtype=torch.bfloat16)
        buffer_atomic_masked[lambda: ((1, 1, 1), (64, 1, 1))](out)
        torch.testing.assert_close(out, torch.full_like(out, 32), rtol=0, atol=0)


def test_shared_atomic_add_returns_old_value():
    if not torch.cuda.is_available() or torch.version.hip is None:
        pytest.skip("AMDGPU required")
    out = torch.empty(64, dtype=torch.int32, device="cuda")
    fetch_add_shared_kernel[lambda: ((1, 1, 1), (64, 1, 1))](out)
    assert torch.equal(out.cpu().sort().values, torch.arange(64, dtype=torch.int32))


@pytest.mark.skipif(not has_rocm(), reason="AMDGPU required")
@pytest.mark.parametrize("aux,scope", [(0, 0), (0, 1), (16, 2)])
def test_buffer_integer_atomics_return_old_values(aux, scope):
    values = torch.tensor([1, -2147483647], dtype=torch.int32, device="cuda")
    counters = torch.zeros(2, dtype=torch.int32, device="cuda")
    old_values = torch.empty(128, dtype=torch.int32, device="cuda")
    buffer_integer_atomics[lambda: ((1, 1, 1), (64, 1, 1))](values, counters, old_values, aux, scope)
    assert counters.cpu().tolist() == [64, -2147483647]
    assert old_values[:64].cpu().sort().values.tolist() == list(range(64))
    assert old_values[64:].cpu().sort().values.tolist() == [-2147483647] * 63 + [0]
