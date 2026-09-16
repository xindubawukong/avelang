"""Static loop expansion and specialization of JIT helpers."""

import ast
from functools import cache
from types import SimpleNamespace

import avelang
import avelang.language as al
import pytest
import torch
from avelang.compiler.static_range import expand_static_ranges
from avelang.testing import has_cuda


def test_nested_static_ranges_preserve_values_and_original_ast():
    source = """
def f():
    total = 0
    for i in al.static_range(3):
        for j in al.static_range(2):
            total += i * 10 + j
    return total, i, j
"""
    module = ast.parse(source)
    before = ast.dump(module)
    expanded = expand_static_ranges(module)
    assert ast.dump(module) == before
    assert not any(isinstance(node, ast.For) for node in ast.walk(expanded))
    scope = {"al": SimpleNamespace(static_range=range)}
    exec(compile(module, "original", "exec"), scope)  # noqa: S102
    expected = scope["f"]()
    exec(compile(expanded, "expanded", "exec"), scope)  # noqa: S102
    assert scope["f"]() == expected == (63, 2, 1)


@pytest.mark.parametrize(
    "body,match",
    [
        ("for i in al.static_range(n):\n        pass", "integer literals"),
        ("for i in al.static_range(257):\n        pass", "256"),
        ("for i in al.static_range(2):\n        i = 3", "reassigned"),
        ("for i in al.static_range(2):\n        break", "break"),
    ],
)
def test_invalid_static_range(body, match):
    with pytest.raises(ValueError, match=match):
        expand_static_ranges(ast.parse("def f():\n    " + body))


@cache
def make_static_helper_kernel(iterations):
    ITERATIONS = iterations

    @avelang.jit
    def accumulate(value: al.u32) -> al.u32:
        total = value + 0
        for i in al.static_range(ITERATIONS):
            total = total * 3 + i
        return total

    @avelang.jit
    def kernel(out: al.Tensor((64,), al.u32)):
        lane = al.thread_id(0)
        out[lane] = accumulate(al.convert(lane, al.u32))

    return kernel


@pytest.mark.skipif(not has_cuda(), reason="Requires a GPU")
def test_constexpr_static_range_in_specialized_jit_helper():
    for iterations in (2, 5):
        out = torch.empty(64, dtype=torch.int32, device="cuda")
        make_static_helper_kernel(iterations)[lambda: ((1, 1, 1), (64, 1, 1))](out)
        expected = torch.arange(64, dtype=torch.int32, device="cuda")
        for i in range(iterations):
            expected = expected * 3 + i
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
