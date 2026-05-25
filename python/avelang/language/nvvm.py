"""NVIDIA-specific language intrinsics."""


def mma_16x8x16_f16_f16(a, b, c):
    pass


def mma_16x8x8_f16_f32(a, b, c):
    pass


def ldmatrix_m8n8_x1_b16(ptr):
    pass


def ldmatrix_m8n8_x1_b16_trans(ptr):
    pass


def ldmatrix_m8n8_x2_b16(ptr):
    pass


def ldmatrix_m8n8_x2_b16_trans(ptr):
    pass


def ldmatrix_m8n8_x4_b16(ptr):
    pass


def ldmatrix_m8n8_x4_b16_trans(ptr):
    pass


def stmatrix_m8n8_x1_b16(ptr, data):
    pass


def stmatrix_m8n8_x1_b16_trans(ptr, data):
    pass


def stmatrix_m8n8_x2_b16(ptr, data):
    pass


def stmatrix_m8n8_x2_b16_trans(ptr, data):
    pass


def stmatrix_m8n8_x4_b16(ptr, data):
    pass


def stmatrix_m8n8_x4_b16_trans(ptr, data):
    pass


def wgmma_fence_aligned():
    pass


def wgmma_group_sync_aligned():
    pass


def wgmma_wait_group_sync(group: int):
    pass


def make_wgmma_descriptor(
    tensor, swizzle_kind: int, l2promo_kind: int, oob_kind: int,
    interleave_kind: int
):
    pass


def wgmma_async(desc_a, desc_b, acc):
    pass


def wgmma_init_accumulator(m: int, n: int):
    pass


def wgmma_store(acc, dst):
    pass


def mbarrier_create():
    pass


def mbarrier_init(barrier, mbar_id: int, count: int = 0,
                  predicate: int = True):
    pass


def mbarrier_try_wait_parity(barrier, parity: int, ticks: int, mbar_id: int):
    pass


def mbarrier_arrive(barrier, mbar_id: int):
    pass


def mbarrier_test_wait(barrier, token, mbar_id: int):
    pass


def mbarrier_arrive_expect_tx(barrier, txcount: int, mbar_id: int,
                              predicate: int):
    pass


def cp_async_ca_shared_global(
    dst, src, dst_offset_bytes, src_offset_bytes, size_bytes: int
):
    pass


def cp_async_commit_group():
    pass


def cp_async_wait_group(n: int):
    pass


def make_tma_descriptor(tensor, smem_layout):
    pass


def tma_fence(desc):
    pass


def tma_load(
    dst, desc, coords, barrier, mbar_id=0, predicate=True,
    multicast_mask=None
):
    pass


def tma_store(src, desc, coords, predicate=True):
    pass
