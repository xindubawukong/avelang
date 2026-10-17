import unittest

import avelang
import avelang.language as S
import torch


@avelang.jit
def conversion_signedness(inp: S.Tensor((2,), S.i32), out: S.Tensor((8,), S.i64)):
    x = inp[0]
    u = S.convert(x, S.u32)
    out[0] = S.convert(-1, S.u64) >> 1
    out[1] = x >> 31
    out[2] = u >> 31
    out[3] = S.convert(u, S.i64)
    out[4] = S.convert(u, S.i32) >> 31
    out[5] = S.convert(x < 0, S.i32)
    out[6] = S.convert(S.convert(x < 0, S.f32), S.i64)
    out[7] = S.convert(S.convert(-1, S.u64), S.i32) >> 31


@avelang.jit
def mixed_integer_order(
    signed: S.Tensor((2,), S.i64),
    unsigned: S.Tensor((2,), S.u32),
    out: S.Tensor((8,), S.i64),
):
    a = signed[0]
    b = unsigned[0]
    out[0] = S.convert((a + b) < 0, S.i64)
    out[1] = S.convert((b + a) < 0, S.i64)
    out[2] = S.convert(a < b, S.i64)
    out[3] = S.convert(b > a, S.i64)
    out[4] = a // b
    out[5] = a % b
    out[6] = S.select(True, a, b) >> 63
    out[7] = S.select(False, b, a) >> 63


@avelang.jit
def literal_operand_order(inp: S.Tensor((2,), S.u64), out: S.Tensor((2,), S.i64)):
    x = inp[0]
    out[0] = (x + 1) >> 62
    out[1] = (1 + x) >> 62


@avelang.jit
def bitwise_extension_policy(inp: S.Tensor((2,), S.i32), out: S.Tensor((2,), S.i64)):
    x = inp[0]
    zero = S.convert(0, S.i64)
    out[0] = x + zero
    out[1] = x | zero


@avelang.jit
def folded_integer_runtime(out: S.Tensor((5,), S.i64)):
    out[0] = 2 + (S.convert(-4, S.i64) % S.convert(3, S.u32))
    out[1] = 1 + ((S.convert(-1, S.u32) + 1) >> 31)
    out[2] = 1 + (S.convert(S.convert(-1, S.u32), S.i64) >> 32)
    out[3] = 1 + (S.convert(-1, S.u64) >> 63)
    out[4] = 1 + ((S.convert(255, S.u8) + S.convert(1, S.u8)) >> 7)


@unittest.skipUnless(torch.cuda.is_available(), "GPU required")
class TestIntegerSignedness(unittest.TestCase):
    def run_kernel(self, kernel, expected, *args):
        out = torch.empty(len(expected), dtype=torch.int64, device="cuda")
        threads = 1
        kernel[lambda: ((1, 1, 1), (threads, 1, 1))](*args, out)
        self.assertEqual(out.tolist(), expected)

    def high_bit_input(self):
        return torch.tensor([-2147483648, 0], dtype=torch.int32, device="cuda")

    def test_runtime_agrees_with_folded_shapes(self):
        # Same expressions as IntegerShapeFoldingMatchesRuntimeTypes in C++.
        self.run_kernel(folded_integer_runtime, [2, 1, 1, 2, 1])

    def test_conversion_and_boolean(self):
        self.run_kernel(
            conversion_signedness,
            [2**63 - 1, -1, 1, 2**31, -1, 1, 1, -1],
            self.high_bit_input(),
        )

    def test_mixed_operands_use_unsigned_common_type(self):
        self.run_kernel(
            mixed_integer_order,
            [0, 0, 0, 0, (2**64 - 5) // 3, (2**64 - 5) % 3, 1, 1],
            torch.tensor([-5, 0], dtype=torch.int64, device="cuda"),
            torch.tensor([3, 0], dtype=torch.int32, device="cuda"),
        )

    def test_literal_operand_order(self):
        self.run_kernel(
            literal_operand_order,
            [2, 2],
            torch.tensor([-(2**63), 0], dtype=torch.int64, device="cuda"),
        )

    def test_bitwise_widening_stays_zero_extended(self):
        self.run_kernel(
            bitwise_extension_policy,
            [-1, 2**32 - 1],
            torch.tensor([-1, 0], dtype=torch.int32, device="cuda"),
        )


if __name__ == "__main__":
    unittest.main()
