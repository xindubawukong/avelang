"""Immutable configuration data stays in the frontend, including across JIT calls."""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass, replace

import _avelang_bindings as _C
import avelang
import avelang.language as al
import pytest
import torch
from avelang.runtime.jit import compute_cache_key, encode_constexpr


BLOCK = 128


@dataclass(frozen=True)
class Tuning:
    enabled: bool = True
    scale: float = 0.5


@dataclass(frozen=True)
class Config:
    m: int = 2
    n: int = 64
    value: int = 7
    tuning: Tuning = Tuning()


@dataclass(frozen=True)
class ExtendedConfig(Config):
    tuning: Tuning = Tuning(scale=0.25)
    unused: int = 11


def test_typed_cache_keys_and_dataclass_fields():
    cache = {}

    def key(value):
        return compute_cache_key(cache, [("constexpr", encode_constexpr(value))], {})

    config = Config()
    assert key(config) == key(replace(config))
    assert key(config) != key(replace(config, value=8))
    assert key(config) != key(replace(config, tuning=Tuning(False)))
    assert key(replace(config, value=True)) != key(replace(config, value=1))
    assert key(replace(config, value=1)) != key(replace(config, value=1.0))
    assert key(replace(config, value=0.0)) != key(replace(config, value=-0.0))
    assert key(al.constexpr(config)) == key(config)


@dataclass
class MutableConfig:
    value: int = 7


@dataclass(frozen=True)
class TupleField:
    value: tuple = (1, 2)


@pytest.mark.parametrize("value", [
    MutableConfig(), (1, 2), Config(tuning=TupleField()),
    Config(tuning=MutableConfig()), Config, 1 << 64, float("nan"),
])
def test_unsupported_config_data(value):
    with pytest.raises(ValueError):
        encode_constexpr(value)


@avelang.jit
def configured_value(x: al.i32, cfg: al.constexpr) -> al.i32:
    def config_value() -> al.i32:
        return cfg.value

    alias = cfg
    tuning = alias.tuning
    if tuning.enabled:
        return x + alias.m * alias.n + config_value()
    return x - alias.m * alias.n + config_value()


@avelang.jit
def forward_config(x: al.i32, cfg: al.constexpr) -> al.i32:
    return configured_value(x, cfg)


@avelang.jit
def configured_tile(shared: al.Tensor((cfg.m, cfg.n), al.i32), cfg: al.constexpr, lane: al.u32) -> al.i32:  # noqa: F821
    return forward_config(shared[0, lane], cfg)


@avelang.jit
def config_kernel(out: al.Tensor((cfg.n,), al.i32), bias: al.i32, cfg: al.constexpr, other: al.constexpr):  # noqa: F821
    lane = al.thread_id(0)
    shared = al.make_shared((cfg.m, cfg.n), al.i32)
    shared[0, lane] = al.convert(lane, al.i32)
    al.syncthreads()
    out[lane] = configured_tile(shared, cfg, lane) + forward_config(100, other) + bias


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_dataclass_helpers_and_launch_cache():
    out = torch.empty(64, dtype=torch.int32, device="cuda")
    configs = [Config(), Config(m=3, value=11, tuning=Tuning(False)), Config()]
    other = ExtendedConfig(m=4, value=13)
    kernels = config_kernel.device_caches[torch.cuda.current_device()][0]
    initial = len(kernels)
    for i, cfg in enumerate(configs):
        config_kernel[lambda: ((1, 1, 1), (64, 1, 1))](out, 1, cfg, other)
        offset = (1 if cfg.tuning.enabled else -1) * cfg.m * cfg.n + cfg.value
        expected = torch.arange(64, dtype=torch.int32, device="cuda") + offset + 100 + 4 * 64 + 13 + 1
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
        assert len(kernels) == initial + min(i + 1, 2)
    with pytest.raises(ValueError):
        config_kernel[lambda: ((1, 1, 1), (64, 1, 1))](out, 1, Config(tuning=MutableConfig()), other)
    assert len(kernels) == initial + 2


@avelang.jit
def config_scale(cfg: al.constexpr) -> al.f64:
    return tuning_scale(cfg.tuning)


@avelang.jit
def tuning_scale(tuning: al.constexpr) -> al.f64:
    return tuning.scale


@avelang.jit
def scalar_fields_kernel(
    floating: al.Tensor((1,), al.f64), integer: al.Tensor((1,), al.i64),
    cfg: al.constexpr, scale: al.f64, offset: al.i64,
):
    floating[0] = config_scale(cfg) + scale
    integer[0] = cfg.value + offset


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_inherited_config_scalar_fields():
    floating = torch.empty(1, dtype=torch.float64, device="cuda")
    integer = torch.empty(1, dtype=torch.int64, device="cuda")
    launch = lambda: ((1, 1, 1), (1, 1, 1))
    cfg = ExtendedConfig(value=1 << 40)
    scalar_fields_kernel[launch](floating, integer, cfg, 0.125, 2)
    assert floating.item() == 0.375
    assert integer.item() == (1 << 40) + 2


def make_captured_helper(value):
    CFG = Config(value=value)

    @avelang.jit
    def captured_helper(x: al.i32, cfg: al.constexpr) -> al.i32:
        return x + CFG.value + cfg.value + BLOCK

    return captured_helper


def make_captured_kernel(value):
    CFG = Config(value=100)
    captured_helper = make_captured_helper(value)
    _ = captured_helper.cache_key

    @avelang.jit
    def captured_kernel(out: al.Tensor((2,), al.i32)):
        BLOCK = 64
        out[0] = captured_helper(3, CFG)
        out[1] = CFG.value + BLOCK

    return captured_kernel


def test_capture_cache_identity():
    assert make_captured_kernel(7).cache_key != make_captured_kernel(9).cache_key


def test_invalid_captured_config():
    config = Config(tuning=MutableConfig())

    @avelang.jit
    def helper() -> al.i32:
        return config.value

    @avelang.jit
    def kernel(out: al.Tensor((1,), al.i32)):
        out[0] = helper()

    for _ in range(2):
        with pytest.raises(ValueError):
            helper._collect_global_constexprs()
        with pytest.raises(ValueError):
            _ = kernel.cache_key
        assert helper.hash is None and kernel.hash is None

    config = Config()
    assert len(kernel.cache_key) == 64


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_helper_uses_its_own_captured_dataclass():
    out = torch.zeros(2, dtype=torch.int32, device="cuda")
    for value in (7, 9):
        make_captured_kernel(value)[lambda: ((1, 1, 1), (1, 1, 1))](out)
        assert out.cpu().tolist() == [103 + value + 128, 100 + 64]


def build_config_ir(body, info=None):
    source = ast.parse(f"""
import avelang.language as al
def kernel(out: al.Tensor((1,), al.i32), cfg: al.constexpr):
    {body}
""")
    generator = _C.MLIRGenerator()
    generator.generate_from_python_ast(ast.Module(body=source.body[:1], type_ignores=[]))
    info = encode_constexpr(Config()) if info is None else info
    generator.visit_function_def(source.body[1], json.dumps([{"name": "cfg", **info}]), "kernel")
    return generator


@pytest.mark.parametrize("body", [
    "out[0] = cfg.missing",
    "cfg.tuning.enabled = False",
    "alias = cfg\n    alias = cfg.tuning",
    "m, n = cfg",
])
def test_invalid_config_access(body):
    with pytest.raises(RuntimeError):
        build_config_ir(body)


def test_cpp_rejects_tuple_constexpr():
    info = {"type": "tuple", "value": [{"type": "i32", "value": 2}]}
    with pytest.raises(RuntimeError):
        build_config_ir("out[0] = 0", info=info)


def test_config_disappears_from_llvm_signature():
    generator = build_config_ir("out[0] = cfg.m + cfg.value")
    ir = generator.get_llvm_ir("amdgcn-amd-amdhsa", "gfx942", 2)
    definition = next(line for line in ir.splitlines() if line.startswith("define ") and "@kernel(" in line)
    assert "ptr" in definition and "i32" not in definition
    assert "store i32 9" in ir
