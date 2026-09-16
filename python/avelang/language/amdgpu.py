"""AMDGPU-specific language intrinsics."""


# Atomic operations


class AtomicScope:
    WORKGROUP = 0
    AGENT = 1
    SYSTEM = 2


def atomic_add(voffset, data, tensor, scope):
    pass


# Matrix instructions


def mfma_16x16x16_f16_f32(a, b, c):
    pass


def mfma_16x16x16_bf16_f32(a, b, c):
    pass


def mfma_f32_16x16x16_bf16(a, b, c):
    pass


def mfma_32x32x8_bf16_f32(a, b, c):
    pass


def mfma_f32_32x32x8_bf16(a, b, c):
    pass


def mfma_scale_16x16x128_fp4(a, scale_a, b, scale_b, c, opsel_a, opsel_b):
    """gfx950 MXFP4 MFMA; packed u32x4 operands and E8M0 scale selectors."""
    pass


# Lane and bit operations


def perm(lhs, rhs, sel):
    pass


def bitreverse(value):
    pass


def get_dpp(old, src, dpp_ctrl, row_mask, bank_mask, bound_ctrl):
    pass


def ds_swizzle(value, pattern):
    """Permute i32 lanes using a constant AMD DS swizzle pattern."""
    pass


def readfirstlane(value):
    pass


# Scalar math


def rcp(value):
    pass


def maximum_f32(a, b):
    """IEEE maximum of f32 values, propagating NaNs."""
    pass


# Buffer memory operations


def make_rsrc(tensor, range_bytes):
    pass


def raw_buffer_load_x1(rsrc, vindex, soffset, aux):
    pass


def raw_buffer_load_x2(rsrc, vindex, soffset, aux):
    pass


def raw_buffer_load_x4(rsrc, vindex, soffset, aux):
    pass


def raw_buffer_load_x1_lds(rsrc, lds_ptr, size, vindex, soffset, offset, aux):
    pass


def raw_buffer_store_x1(vdata, rsrc, vindex, soffset, aux):
    pass


def raw_buffer_store_x2(vdata, rsrc, vindex, soffset, aux):
    pass


def raw_buffer_store_x4(vdata, rsrc, vindex, soffset, aux):
    pass


# Packed conversions


def cvt_pk_fp8_f32(src0, src1, old, word_sel):
    pass


def cvt_pk_bf8_f32(src0, src1, old, word_sel):
    pass


def cvt_pk_f32_bf8(src, word_sel):
    pass


def cvt_scalef32_pk_fp4_f32(old, a, b, scale, byte_sel):
    """Quantize two f32 values into the selected byte of a packed FP4 word."""
    pass


# Synchronization and scheduling


def s_waitcnt(vmcnt, expcnt, lgkmcnt):
    pass


def sched_group_barrier(mask, size, group_id):
    pass


def sched_barrier(mask):
    pass


def s_setprio(priority):
    pass


def v_setvskip(mask, skip_id):
    pass
