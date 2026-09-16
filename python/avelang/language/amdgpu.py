"""AMDGPU-specific language intrinsics."""


# Atomic operations


class AtomicScope:
    WORKGROUP = 0
    AGENT = 1
    SYSTEM = 2


def atomic_add(voffset, data, tensor, scope):
    """Add at a byte offset in global/LDS storage and return the old value."""


def raw_buffer_atomic_add_u32(value, resource, byte_offset, soffset, aux):
    """Return old u32 and add value; aux 0 selects agent, 16 system scope."""


def raw_buffer_atomic_or_u32(value, resource, byte_offset, soffset, aux):
    """Return old u32 and OR value; aux 0 selects agent, 16 system scope."""


def raw_buffer_atomic_add_bf16x2(value, resource, byte_offset):
    """Add a BF16 pair; a buffer offset outside the resource is discarded."""
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
    """Load one dword (4 bytes) per lane into LDS; size must be 4."""
    pass


def raw_buffer_load_x4_lds(rsrc, lds_ptr, size, vindex, soffset, offset, aux):
    """Load four dwords (16 bytes) per lane into LDS; size must be 16."""
    pass


def raw_buffer_store_u8(value, resource, byte_offset, scalar_offset, aux):
    """Store one byte through a raw buffer resource."""
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


def compiler_barrier():
    """Prevent compiler motion of memory operations across this point."""


def fence(ordering, scope):
    """Ordering: 0 acquire, 1 release, 2 both; scope: 0 workgroup, 1 agent, 2 system."""


def sched_group_barrier(mask, size, group_id):
    pass


def sched_barrier(mask):
    pass


def s_setprio(priority):
    pass


def s_sleep(cycles):
    """Back off a polling wave; constant hardware sleep immediate in [0, 15]."""


def v_setvskip(mask, skip_id):
    pass
