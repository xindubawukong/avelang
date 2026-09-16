#!/usr/bin/env python3
import unittest

import avelang
import avelang.language as S
import pytest
import torch

GLOBAL_SIZE = 16
GLOBAL_OFFSET = S.constexpr(3)
GLOBAL_F32_EXACT = -0.5


@avelang.jit
def add_global(x: S.i32) -> S.i32:
    return x + GLOBAL_OFFSET


@avelang.jit
def kernel_global_constexpr(
    input_data: S.Tensor((GLOBAL_SIZE,), S.i32),
    output_data: S.Tensor((GLOBAL_SIZE,), S.i32),
):
    shared_buf = S.make_shared((GLOBAL_SIZE,), S.i32)
    tid = S.thread_id(0)
    shared_buf[tid] = add_global(input_data[tid])
    S.syncthreads()
    output_data[tid] = shared_buf[GLOBAL_SIZE - 1 - tid]


@avelang.jit
def kernel_global_exact_float_constant(
    output_data: S.Tensor((4,), S.f32),
):
    shared_buf = S.make_shared((4,), S.f32)
    tid = S.thread_id(0)
    shared_buf[tid] = GLOBAL_F32_EXACT
    S.syncthreads()
    output_data[tid] = shared_buf[tid]


class TestGlobalConstexprInjection(unittest.TestCase):
    def test_global_constexpr_injection(self):
        input_data = torch.arange(GLOBAL_SIZE, dtype=torch.int32, device="cuda")
        output_data = torch.zeros((GLOBAL_SIZE,), dtype=torch.int32, device="cuda")

        expected = (input_data + GLOBAL_OFFSET.value).flip(0)

        kernel_global_constexpr[lambda: ((1, 1, 1), (GLOBAL_SIZE, 1, 1))](input_data, output_data)

        actual = output_data.cpu()
        expected = expected.cpu()

        self.assertTrue(
            torch.equal(actual, expected),
            f"Expected: {expected.tolist()}, Actual: {actual.tolist()}",
        )

    def test_exact_float_constant_can_implicitly_demote(self):
        output_data = torch.zeros((4,), dtype=torch.float32, device="cuda")
        expected = torch.full((4,), GLOBAL_F32_EXACT, dtype=torch.float32, device="cuda")

        kernel_global_exact_float_constant[lambda: ((1, 1, 1), (4, 1, 1))](output_data)

        self.assertTrue(
            torch.equal(output_data.cpu(), expected.cpu()),
            f"Expected: {expected.tolist()}, Actual: {output_data.tolist()}",
        )


def make_offset_helper(offset):
    @avelang.jit
    def add_offset(values: S.Tensor((4,), S.u32), index: S.u32) -> S.u32:
        return values[index] + offset

    return add_offset


def make_distinct_capture_kernel():
    offset = 100
    add_offset = make_offset_helper(7)

    @avelang.jit
    def kernel(output: S.Tensor((2,), S.u32)):
        shared = S.make_shared((4,), S.u32)
        shared[0] = S.convert(3, S.u32)
        output[0] = add_offset(shared, S.convert(0, S.u32))
        output[1] = S.convert(offset, S.u32)

    return kernel


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_helper_capture_is_not_replaced_by_same_named_caller_capture():
    output = torch.zeros(2, dtype=torch.int32, device="cuda")
    make_distinct_capture_kernel()[lambda: ((1, 1, 1), (64, 1, 1))](output)
    assert output.cpu().tolist() == [10, 100]


def make_shadow_kernel():
    offset = 100

    @avelang.jit
    def local_offset(values: S.Tensor((2,), S.u32)) -> S.u32:
        offset = values[0]
        return offset + values[1]

    @avelang.jit
    def kernel(values: S.Tensor((2,), S.u32), output: S.Tensor((2,), S.u32)):
        if S.thread_id(0) == 0:
            output[0] = local_offset(values)
            output[1] = S.convert(offset, S.u32)

    return kernel


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_caller_capture_does_not_make_helper_local_immutable():
    values = torch.tensor([3, 7], dtype=torch.int32, device="cuda")
    output = torch.zeros(2, dtype=torch.int32, device="cuda")
    make_shadow_kernel()[lambda: ((1, 1, 1), (64, 1, 1))](values, output)
    assert output.cpu().tolist() == [10, 100]

@avelang.jit
def large_literals_kernel(output: S.Tensor((2,), S.u64)):
    if S.thread_id(0) == 0:
        output[0] = S.convert(0x100000003, S.u64)
        output[1] = S.convert(3086034172, S.u64)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_large_integer_literals_keep_their_bits():
    output = torch.zeros(2, dtype=torch.int64, device="cuda")
    large_literals_kernel[lambda: ((1, 1, 1), (64, 1, 1))](output)
    assert output.cpu().tolist() == [0x100000003, 3086034172]

if __name__ == "__main__":
    unittest.main()
