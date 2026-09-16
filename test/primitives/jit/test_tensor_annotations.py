"""Factory dimensions used only in annotations survive nested JIT calls."""

import ast

import avelang
import avelang.language as al
import pytest
import torch


def make_helpers(size):
    @avelang.jit
    def leaf(values: al.Tensor((size,), al.u32), index: al.u32) -> al.u32:
        return values[index]

    @avelang.jit
    def middle(values: al.Tensor((size,), al.u32), index: al.u32) -> al.u32:
        return leaf(values, index)

    @avelang.jit
    def kernel(values: al.Tensor((size,), al.u32), output: al.Tensor((2,), al.u32), index: al.u32):
        output[0] = middle(values, index)

    return leaf, middle, kernel


def test_annotation_only_factory_dimensions_are_materialized_and_hashed():
    a, _, _ = make_helpers(5)
    b, _, _ = make_helpers(9)
    assert "size" not in a.get_capture_scope()
    assert a.cache_key != b.cache_key
    assert ast.literal_eval(a.parse().body[0].args.args[0].annotation.args[0]) == (5,)
    assert ast.literal_eval(b.parse().body[0].args.args[0].annotation.args[0]) == (9,)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@pytest.mark.parametrize("size", [5, 9])
def test_nested_tensor_helpers_keep_factory_dimensions(size):
    _, _, kernel = make_helpers(size)
    values = torch.arange(size, dtype=torch.int32, device="cuda") + 100
    output = torch.zeros(2, dtype=torch.int32, device="cuda")
    kernel[lambda: ((1, 1, 1), (64, 1, 1))](values, output, size - 1)
    assert output.cpu().tolist() == [99 + size, 0]
