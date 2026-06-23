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


def cp_async_bulk_commit_group():
    pass


def cp_async_bulk_global_shared_cta(
    dst, src, size, dst_offset_bytes=0, src_offset_bytes=0,
    l2_cache_hint=None, byte_mask=None
):
    pass


def cp_async_bulk_prefetch(src, size, src_offset_bytes=0,
                           l2_cache_hint=None):
    pass


def cp_async_bulk_shared_cluster_global(
    dst, src, mbar, size, dst_offset_bytes=0, src_offset_bytes=0,
    mbar_offset_bytes=0, multicast_mask=None, l2_cache_hint=None
):
    pass


def cp_async_bulk_shared_cluster_shared_cta(
    dst, src, mbar, size, dst_offset_bytes=0, src_offset_bytes=0,
    mbar_offset_bytes=0
):
    pass


def cp_async_bulk_tensor_global_shared_cta(
    desc, src, coords, src_offset_bytes=0, l2_cache_hint=None,
    predicate=None
):
    pass


def cp_async_bulk_tensor_prefetch(
    desc, coords, im2col_offsets=None, l2_cache_hint=None
):
    pass


def cp_async_bulk_tensor_reduce(
    desc, src, coords, red_kind: int, src_offset_bytes=0, l2_cache_hint=None
):
    pass


def cp_async_bulk_tensor_shared_cluster_global(
    dst, desc, coords, mbar, dst_offset_bytes=0, mbar_offset_bytes=0,
    im2col_offsets=None, multicast_mask=None, l2_cache_hint=None,
    predicate=None
):
    pass


def cp_async_bulk_wait_group(group: int, read: bool = False):
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
