---
layout: layouts/base.njk
title: Builtin Types
description: Scalar, tensor, pointer, constexpr, and dynamic types in Ave.
---

# Builtin Types

All types in Ave are under the `avelang.language` package. For simplicity, programs often import the package as `al`:

```python
import avelang.language as al
```

## Scalar Types

For value types, Ave supports signed integers (`al.i8`, `al.i16`, `al.i32`, `al.i64`), unsigned integers (`al.u8`, `al.u16`, `al.u32`, `al.u64`), and floating point values (`al.f16`, `al.bf16`, `al.f32`, `al.f64`). It also supports the `al.void` type to denote void type.

Value conversion is described in [Builtin Functions](../builtin-functions/).

## Tensor Types

`al.Tensor(shape, type)` denotes the type of a tensor. A tensor is a multidimensional view over a memory region. The shape describes the logical dimensions of the tensor, and the type describes the value type of each element. For example, `al.Tensor((16, 16), al.f32)` denotes a two-dimensional tensor with 16 rows and 16 columns of 32-bit floating point values.

```python
al.Tensor((16, 16), al.f32)
```

Tensors are indexed by their logical coordinates. A two-dimensional tensor can be indexed with two coordinates:

```python
x[row, col]
```

For contiguous tensors, strides are inferred and can be omitted. A row-major `al.Tensor((16, 16), al.f32)` has inferred strides `(16, 1)`, so `x[row, col]` maps to the linear element index `row * 16 + col`.

`al.Tensor(shape, stride, type)` denotes a tensor type with explicit strides. If the shape is `(s0, s1, ...)` and the stride is `(t0, t1, ...)`, subscript coordinates `(c0, c1, ...)` map to a linear element index as `c0 * t0 + c1 * t1 + ...`.

```python
al.Tensor((s0, s1, ...), (t0, t1, ...), dtype)
```

For example, `al.Tensor((16, 16), (1, 16), al.f32)` denotes a column-major `16 x 16` tensor because moving by one row advances by one element and moving by one column advances by 16 elements.

```python
al.Tensor((16, 16), (1, 16), al.f32)
```

### Packed And Nested Shapes

Tensor shapes may be nested to describe packed, hierarchical, or hardware-specific layouts. This is common when vectorizing memory access.

A contiguous BF16 row with 128 elements can be represented as 16 vectors, each vector holding four `i32` words:

```python
al.Tensor((128, 16, 4), al.i32)
```

Each group of four `i32` words is 16 bytes. Interpreted as BF16 storage, those same 16 bytes contain eight BF16 values. A view with this shape changes the logical interpretation of the same memory; it does not copy or rearrange data.

This kind of view exposes a row as:

```text
(row, vector, word)
```

where `vector` selects a contiguous 16-byte chunk and `word` selects one of the four `i32` words in that chunk.

## Pointer Types

`al.Pointer(dtype)` marks a raw pointer argument. Use it when the logical shape or stride is known at runtime.

```python
@avelang.jit
def fill(x_ptr: al.Pointer(al.i32), n: al.u32, value: al.i32):
    ...
```

The pointer type describes the element type at the raw pointer. `al.make_tensor` combines the pointer with a runtime layout to create an indexable tensor view.

Dynamic dimensions are usually introduced by accepting a raw pointer and runtime shape values, then constructing a tensor view with `al.make_layout`. The pointer describes the element type, and the layout describes how runtime coordinates map to memory.

For a one-dimensional buffer, the runtime length is used directly as the layout shape:

```python
@avelang.jit
def fill(x_ptr: al.Pointer(al.i32), n: al.u32, value: al.i32):
    x = al.make_tensor(x_ptr, al.i32, al.make_layout((n,), (1,)))
```

For a row-major matrix, the row stride is the runtime width:

```python
A = al.make_tensor(A_ptr, al.bf16, al.make_layout((m, k), (k, 1)))
```

This casts `A_ptr` to an indexable `(m, k)` BF16 tensor where `A[row, col]` maps to `row * k + col`.

The tensor can then be cast again with `al.view` to expose a packed layout. For example, when `k` is divisible by eight BF16 values, the same row-major matrix can be viewed as packed 16-byte vectors:

```python
k_vecs = k >> 3
packed_row_stride = k >> 1

A_vec = al.view(
    A,
    al.i32,
    al.make_layout((m, k_vecs, 4), (packed_row_stride, 4, 1)),
)
```

Here `A_vec[row, vec]` is four contiguous `i32` words, representing eight BF16 values from the original row. This is the common pattern for runtime-shape kernels: use `al.Pointer` for the raw argument, `al.make_layout` and `al.make_tensor` to recover the logical tensor, and `al.view` to cast that tensor into the layout needed by vectorized memory access or hardware intrinsics.

## Constexpr

`al.constexpr` marks specialization-time parameters. These values are known to the compiler and can be used in shapes, shared-memory allocations, and loop bounds.

```python
@avelang.jit
def tiled(x: al.Pointer(al.i32), BLOCK: al.constexpr):
    smem = al.make_shared((BLOCK,), al.i32)
```

Parameters may also be frozen dataclass instances, recursively containing
`bool`, signed 64-bit `int`, `float`, and frozen dataclasses. This lets a
kernel and its JIT helpers share a single configuration:

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    m: int = 16
    n: int = 64
    enabled: bool = True

@avelang.jit
def helper(x: al.i32, cfg: al.constexpr) -> al.i32:
    if cfg.enabled:
        return x + cfg.m * cfg.n
    return x

@avelang.jit
def configured(out: al.Tensor((1,), al.i32), cfg: al.constexpr):
    smem = al.make_shared((cfg.m, cfg.n), al.i32)
    out[0] = helper(0, cfg)
```

Fields are resolved in the compiler frontend. Configuration objects are read-only
and do not become runtime kernel arguments; local configuration aliases cannot be
rebound. Passing a different configuration selects another compiled specialization;
equivalent contents reuse the same JIT cache entry. Dataclass construction and
methods are not supported inside JIT functions.

For Tensor parameter shapes that reference a configuration parameter, add
`from __future__ import annotations` at the top of the file:

```python
@avelang.jit
def helper(x: al.Tensor((cfg.m, cfg.n), al.i32), cfg: al.constexpr):
    x[cfg.m - 1, cfg.n - 1] = 1
```

Tuples are not supported as constexpr parameters or dataclass fields; ordinary
tuple expressions such as the shared-memory shape above still work.
