"""Public PyTorch API and kernel dispatch for AMDGPU FP4 GEMM."""

from functools import cache

import torch

from .config import FP4GemmConfig
from .kernel import _make_fp4_gemm_kernel
from .solution import (
    SOLUTION_LIST,
    MatmulElementB,
    MatmulFeatures,
    MatmulMfmaType,
    MatmulWarpPartition,
    SolutionId,
    choose_default_solution,
)
from .utils import (
    SCALE_TILE_N,
    WEIGHT_TILE_K,
    packed_scale_shape,
    packed_weight_shape,
)

SUPPORTED_SOLUTIONS = frozenset(
    solution
    for solution in SOLUTION_LIST
    if solution.features in (MatmulFeatures.GRID, MatmulFeatures.GRID | MatmulFeatures.HIGH_PRECISION)
    and (solution.element_b, solution.mfma_type) in (
        (MatmulElementB.NVFP4, MatmulMfmaType.FP16),
        (MatmulElementB.NVFP4, MatmulMfmaType.BF16),
        (MatmulElementB.MXFP4, MatmulMfmaType.BF16),
    )
    and solution.warp_partition == MatmulWarpPartition.NK
)


def _ceildiv(lhs: int, rhs: int) -> int:
    return (lhs + rhs - 1) // rhs


def resolve_solution(
    m: int,
    n: int,
    k: int,
    solution_id: SolutionId | int | None = None,
    *,
    element_b: MatmulElementB = MatmulElementB.NVFP4,
    mfma_type: MatmulMfmaType = MatmulMfmaType.BF16,
    high_precision: bool | None = None,
) -> SolutionId:
    if solution_id is None or solution_id == -1:
        solution = choose_default_solution(
            m, n, k, element_b=element_b, mfma_type=mfma_type,
            high_precision=bool(high_precision),
        )
    elif isinstance(solution_id, SolutionId):
        solution = solution_id
    else:
        solution = SolutionId.from_int(solution_id)
    if solution not in SUPPORTED_SOLUTIONS:
        raise ValueError(f"unsupported AveLang FP4 solution {int(solution):#x}")
    if solution.element_b != element_b or solution.mfma_type != mfma_type:
        raise ValueError("solution_id does not match the input dtype and FP4 format.")
    if high_precision is not None and bool(solution.features & MatmulFeatures.HIGH_PRECISION) != high_precision:
        raise ValueError("solution_id does not match high_precision.")
    if n % solution.group_n or k % solution.group_k:
        raise ValueError(
            f"solution {int(solution):#x} is incompatible with N={n}, K={k}"
        )
    return solution


@cache
def get_fp4_gemm_kernel(solution_id: int, arch: str = "gfx942"):
    solution = SolutionId.from_int(solution_id)
    if solution not in SUPPORTED_SOLUTIONS:
        raise ValueError(f"unsupported AveLang FP4 solution {solution_id:#x}")
    return _make_fp4_gemm_kernel(solution, arch)


def fp4_gemm(
    A: torch.Tensor,
    packed_weight: torch.Tensor,
    packed_scales: torch.Tensor,
    global_scale: torch.Tensor | float,
    out: torch.Tensor | None = None,
    solution_id: SolutionId | int | None = None,
    *,
    element_b: MatmulElementB = MatmulElementB.NVFP4,
    high_precision: bool | None = None,
) -> torch.Tensor:
    """Multiply FP16/BF16 ``A`` by an AveLang-packed FP4 matrix.

    ``packed_weight`` and ``packed_scales`` must come from
    :func:`avelang_kernels.amdgpu.fp4_gemm.utils.repack_fp4` and
    :func:`avelang_kernels.amdgpu.fp4_gemm.utils.process_fp4_scales`,
    respectively. Pass the same ``element_b`` to ``process_fp4_scales``
    and this function. MXFP4 requires BF16 inputs
    and uint8 E8M0 source scales (32 weights per scale); NVFP4 uses E4M3
    source scales (16 weights per scale). Output has A's dtype.

    ``high_precision`` restores weights before MFMA to reduce dequantization
    underflow. If omitted, an explicit solution_id controls this flag; default
    dispatch uses normal precision. Native MFMA/output dtype limits still apply.
    """
    if (
        A.ndim != 2
        or A.dtype not in (torch.float16, torch.bfloat16)
        or not A.is_contiguous()
        or A.device.type != "cuda"
    ):
        raise ValueError("A must be a contiguous 2-D CUDA FP16/BF16 tensor.")
    if element_b not in (MatmulElementB.NVFP4, MatmulElementB.MXFP4):
        raise ValueError(f"unsupported FP4 format: {element_b}")
    if element_b == MatmulElementB.MXFP4 and A.dtype != torch.bfloat16:
        raise ValueError("MXFP4 requires BF16 inputs.")
    mfma_type = MatmulMfmaType.FP16 if A.dtype == torch.float16 else MatmulMfmaType.BF16
    m, k = A.shape
    if k % WEIGHT_TILE_K:
        raise ValueError(f"A's K dimension must be divisible by {WEIGHT_TILE_K}.")
    if (
        packed_weight.ndim != 5
        or packed_weight.dtype != torch.int32
        or not packed_weight.is_contiguous()
        or packed_weight.shape[1] != 4
        or packed_weight.shape[3:] != (64, 4)
    ):
        raise ValueError("packed_weight is not an AveLang packed FP4 weight tensor.")
    n = packed_weight.shape[2] * 32
    if n % SCALE_TILE_N or packed_weight.shape != packed_weight_shape(n, k):
        raise ValueError(
            f"packed_weight must have AveLang packed shape {packed_weight_shape(n, k)}."
        )
    if (
        packed_scales.dtype != torch.uint8
        or packed_scales.shape != packed_scale_shape(n, k, element_b=element_b)
        or not packed_scales.is_contiguous()
    ):
        raise ValueError(
            f"packed_scales must have AveLang packed uint8 shape "
            f"{packed_scale_shape(n, k, element_b=element_b)}."
        )
    if not (A.device == packed_weight.device == packed_scales.device):
        raise ValueError("All inputs must be on the same device.")

    if isinstance(global_scale, torch.Tensor):
        if (
            global_scale.dtype != torch.float32
            or global_scale.numel() != 1
            or global_scale.device != A.device
        ):
            raise ValueError(
                "global_scale must contain one float32 value on A's device."
            )
        scale_tensor = global_scale.reshape(1)
    else:
        scale_tensor = torch.tensor(
            [float(global_scale)], dtype=torch.float32, device=A.device
        )

    solution = resolve_solution(
        m, n, k, solution_id, element_b=element_b,
        mfma_type=mfma_type, high_precision=high_precision,
    )
    config = FP4GemmConfig.from_solution(solution)
    if out is None:
        out = torch.empty((m, n), dtype=A.dtype, device=A.device)
    elif (
        out.shape != (m, n)
        or out.dtype != A.dtype
        or out.device != A.device
        or not out.is_contiguous()
    ):
        raise ValueError(
            f"out must be contiguous {A.dtype} with shape {(m, n)} on {A.device}."
        )

    grid_m = _ceildiv(m, config.group_m)
    grid_n = n // config.group_n
    arch = torch.cuda.get_device_properties(A.device).gcnArchName.split(":")[0]
    kernel = get_fp4_gemm_kernel(int(solution), arch)
    kernel[lambda: ((grid_m, grid_n, 1), (config.threads, 1, 1))](
        A,
        packed_weight,
        packed_scales,
        scale_tensor,
        out,
        m,
        n,
        k,
        num_warps=config.num_warps,
    )
    return out
