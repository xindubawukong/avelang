---
layout: layouts/base.njk
title: Installation guide
---

## Prerequisites

- LLVM, MLIR, and Clang 22 or 24, all from the same toolchain. The ROCm backend is tested with LLVM 22.1.8 and the ROCm LLVM 24 development toolchain. CUDA validation still targets LLVM 22.
- CMake 3.25 or newer
- Ninja
- Python 3.10 or newer
- CUDA for NVIDIA targets, or ROCm for AMDGPU targets

## Building and installing Ave

We recommend building Ave inside a [uv](https://docs.astral.sh/uv) Python virtual environment, and using Clang for faster compilation. Ave supports both CUDA and ROCm backends through the `AVE_LANG_BACKEND` CMake option. Valid values are `cuda`, `rocm`, or `cuda;rocm`. For example, to enable the CUDA backend, run the following commands from the source directory (replace `/path/to/llvm` with your LLVM install prefix, such as `/usr/local` in our CUDA environment):

```bash
uv venv .venv
source .venv/bin/activate
CMAKE_ARGS='-DAVE_LANG_BACKEND=cuda -DWITH_PYTHON=ON \
  -DCMAKE_C_COMPILER=/path/to/llvm22/bin/clang \
  -DCMAKE_CXX_COMPILER=/path/to/llvm22/bin/clang++ \
  -DCMAKE_PREFIX_PATH=/path/to/llvm22' uv pip install -e .
```

For ROCm:

Use the prefix of your LLVM 22 or 24 toolchain in place of `/path/to/llvm22`.
Include `/opt/rocm` separately so CMake can find HIP and the device libraries.

```bash
uv venv .venv
source .venv/bin/activate
CMAKE_ARGS="-DAVE_LANG_BACKEND=rocm -DWITH_PYTHON=ON \
  -DCMAKE_C_COMPILER=/path/to/llvm22/bin/clang \
  -DCMAKE_CXX_COMPILER=/path/to/llvm22/bin/clang++ \
  -DCMAKE_PREFIX_PATH=/path/to/llvm22;/opt/rocm" uv pip install -e .
```

Ave uses Clang from the selected LLVM toolchain for AMDGPU device linking.
This prevents ROCm's bundled Clang from trying to read newer LLVM bitcode.
For a custom installation, set `-DAVE_LANG_AMDGPU_CLANG_EXECUTABLE=/path/to/matching/clang`.
Rebuild the Python extension when switching LLVM versions; each build uses one
LLVM version and regenerates its embedded MLIR bytecode with that version.

You should be able to run the python test via `python -m pytest test` once the build and installation are completed.

## Development build

To avoid rebuilding the full package on every `pip install` during development, configure a separate build directory and write the Python extension directly into `python/`. For a CUDA development build:

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
cmake -S . -B build -GNinja -DCMAKE_BUILD_TYPE=Release \
  -DAVE_LANG_BACKEND=cuda \
  -DWITH_PYTHON=ON \
  -DCMAKE_C_COMPILER=/path/to/llvm/bin/clang \
  -DCMAKE_CXX_COMPILER=/path/to/llvm/bin/clang++ \
  -DCMAKE_PREFIX_PATH=/path/to/llvm \
  -DAVE_LANG_PYTHON_LIBRARY_OUTPUT_DIRECTORY="$(pwd)/python"
ninja -C build
ctest --test-dir build --output-on-failure
PYTHONPATH=python python -m pytest test
```

For ROCm, switch `AVE_LANG_BACKEND` to `rocm`, set `CMAKE_PREFIX_PATH` to `'/path/to/llvm;/opt/rocm'`, and run `PYTHONPATH=python python -m pytest test`.

Running `ninja -C build` enables incremental builds of Ave to speed up development.

After installation, continue with the [tutorials](../../tutorials/) to write and launch Ave kernels.
