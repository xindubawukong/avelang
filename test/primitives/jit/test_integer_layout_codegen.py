"""Check address extension without allocating buffers with huge strides."""

import ast
import re

import _avelang_bindings as C
import pytest
import torch


@pytest.mark.skipif(not torch.version.hip, reason="AMDGPU code generation")
@pytest.mark.parametrize("dtype,instruction", [("u32", "zext"), ("i32", "sext")])
def test_dynamic_layout_integer_extension(dtype, instruction):
    source = f"""
def layout_kernel(p: S.Pointer(S.i32), value: S.{dtype}, out: S.Tensor((2,), S.i32)):
    t = S.make_tensor(p, S.i32, S.make_layout((2, 2), (value, 1)))
    out[0] = t[1, 0]
"""
    generator = C.MLIRGenerator()
    generator.generate_from_python_ast(ast.parse("import avelang\nimport avelang.language as S"))
    generator.visit_function_def(ast.parse(source).body[0], "[]", "kernel")
    llvm_ir = generator.get_llvm_ir("amdgcn-amd-amdhsa", "gfx90a", 0)
    assert re.search(rf"\b{instruction} i32 %\w+ to i64", llvm_ir), llvm_ir
    opposite = "sext" if instruction == "zext" else "zext"
    assert not re.search(rf"\b{opposite} i32 %\w+ to i64", llvm_ir), llvm_ir
