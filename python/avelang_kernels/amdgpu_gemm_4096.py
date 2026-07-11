import torch

import avelang
import avelang.language as al


WARP_SIZE = 64
NUM_WARPS = 4
M_SIZE = 4096
N_SIZE = 4096
K_SIZE = 4096
GROUP_M = 224
GROUP_N = 256
GROUP_K = 64
M_GROUPS = (M_SIZE + GROUP_M - 1) // GROUP_M
N_GROUPS = (N_SIZE + GROUP_N - 1) // GROUP_N
K_TILES = K_SIZE // GROUP_K
GRID_SIZE = M_GROUPS * N_GROUPS
WARP_PER_ROW = 2
WARP_PER_COL = 2
WARP_MAT_M = GROUP_M // WARP_PER_ROW
WARP_MAT_N = GROUP_N // WARP_PER_COL
M_TILES_PER_WARP = WARP_MAT_M // 16
N_TILES_PER_WARP = WARP_MAT_N // 16
VEC_SIZE = 8
THREADS = WARP_SIZE * NUM_WARPS
REG_ROWS_A = GROUP_M * GROUP_K // VEC_SIZE // THREADS
REG_ROWS_B = GROUP_N * GROUP_K // VEC_SIZE // THREADS
GLOBAL_WORDS_PER_ROW = GROUP_K * 2 // 4
GLOBAL_ROWS_PER_ROUND = THREADS // GLOBAL_WORDS_PER_ROW
REG_WORDS_A = GROUP_M // GLOBAL_ROWS_PER_ROUND
REG_WORDS_B = GROUP_N // GLOBAL_ROWS_PER_ROUND
BF16_BYTES = 2
NUM_BATCH_K = 4
SHM_READ_ROWS_A = 1
SHM_READ_ROWS_B = 8
SHM_PAD_BYTES = 8
SHM_PAD_WORDS = SHM_PAD_BYTES // 4
SHM_ROW_WORDS = GROUP_K * BF16_BYTES // 4
SHM_GROUP_WORDS_A = SHM_READ_ROWS_A * SHM_ROW_WORDS + SHM_PAD_WORDS
SHM_GROUP_WORDS_B = SHM_READ_ROWS_B * SHM_ROW_WORDS + SHM_PAD_WORDS
SHM_GROUPS_A = GROUP_M // SHM_READ_ROWS_A
SHM_GROUPS_B = GROUP_N // SHM_READ_ROWS_B
SHM_TOTAL_WORDS_A = SHM_GROUPS_A * SHM_GROUP_WORDS_A
SHM_TOTAL_WORDS_B = SHM_GROUPS_B * SHM_GROUP_WORDS_B
SHM_TOTAL_BF16_A = SHM_TOTAL_WORDS_A * 2
SHM_TOTAL_BF16_B = SHM_TOTAL_WORDS_B * 2
SHM_CHUNKS_PER_ROW = GROUP_K // VEC_SIZE

MI300_CU_COUNT = 38 * 8
WGM_XCC = 8
WORKGROUP_MAPPING = 8
STAGGER_U_MASK = 0
STAGGER_U_STRIDE = 0
STAGGER_U_MAPPING = 1

SCHED_MASK_MFMA = 0x8
SCHED_MASK_BUFFER_LOAD = 0x20
SCHED_MASK_DS_READ = 0x100
SCHED_MASK_DS_WRITE = 0x200


@avelang.jit
def _wgm_mapping() -> (al.u32, al.u32):
    linear_group_id = al.block_id(0)
    m_groups = al.convert(M_GROUPS, al.u32)
    n_groups = al.convert(N_GROUPS, al.u32)
    total_groups = al.convert(GRID_SIZE, al.u32)

    cu_count = al.convert(MI300_CU_COUNT, al.u32)
    wgm_xcc = al.convert(WGM_XCC, al.u32)
    workgroup_mapping = al.convert(WORKGROUP_MAPPING, al.u32)

    linear_group_limit = (total_groups // wgm_xcc) * wgm_xcc
    cu_base = (linear_group_id // cu_count) * cu_count
    cu_xcc = (linear_group_id % cu_count) // wgm_xcc
    cu_base = cu_base + cu_xcc

    cu_tail_limit = (total_groups // cu_count) * cu_count
    active_cu = (
        (total_groups % cu_count) if (linear_group_id > cu_tail_limit) else cu_count
    )
    cu_xcc_stride = (active_cu // wgm_xcc) * (linear_group_id % wgm_xcc)
    linear_group_mapped = cu_base + cu_xcc_stride

    linear_group_id = (
        linear_group_mapped
        if (linear_group_id < linear_group_limit)
        else linear_group_id
    )

    group_m = linear_group_id // n_groups
    group_n = linear_group_id - group_m * n_groups

    mapping_block = group_m // workgroup_mapping
    mapping_linear = group_n + (group_m % workgroup_mapping) * n_groups
    mapping_groups = m_groups // workgroup_mapping
    mapping_tail = m_groups % workgroup_mapping
    mapping_tail = workgroup_mapping if (mapping_tail == 0) else mapping_tail

    mapping_span = (
        mapping_tail if (mapping_block >= mapping_groups) else workgroup_mapping
    )

    group_n = mapping_linear // mapping_span
    group_m = mapping_linear % mapping_span
    group_m = group_m + mapping_block * workgroup_mapping

    return group_m, group_n


@avelang.jit
def _staggered_k_tile(
    group_m: al.u32, group_n: al.u32, k_total: al.u32, offset: al.u32
) -> al.u32:
    return offset


@avelang.jit
def _load_global_a(
    src_rsrc: al.Tensor((4,), al.u32),
    group_row: al.u32,
    k_idx: al.u32,
    tid: al.u32,
    reg: al.Tensor((REG_WORDS_A,), al.u32),
):
    row = tid // GLOBAL_WORDS_PER_ROW
    col_word = tid - row * GLOBAL_WORDS_PER_ROW
    thread_offset = row * K_SIZE * BF16_BYTES + col_word * 4
    tile_offset = (group_row * GROUP_M * K_SIZE + k_idx * GROUP_K) * BF16_BYTES
    thread_offset_stride = GLOBAL_ROWS_PER_ROUND * K_SIZE * BF16_BYTES

    for i in al.range(REG_WORDS_A):
        reg[i] = al.amdgpu.raw_buffer_load_x1(
            src_rsrc,
            thread_offset,
            tile_offset + i * thread_offset_stride,
            0,
        )


@avelang.jit
def _load_global_b(
    src_rsrc: al.Tensor((4,), al.u32),
    group_row: al.u32,
    k_idx: al.u32,
    tid: al.u32,
    reg: al.Tensor((REG_WORDS_B,), al.u32),
):
    row = tid // GLOBAL_WORDS_PER_ROW
    col_word = tid - row * GLOBAL_WORDS_PER_ROW
    thread_offset = row * K_SIZE * BF16_BYTES + col_word * 4
    tile_offset = (group_row * GROUP_N * K_SIZE + k_idx * GROUP_K) * BF16_BYTES
    thread_offset_stride = GLOBAL_ROWS_PER_ROUND * K_SIZE * BF16_BYTES

    for i in al.range(REG_WORDS_B):
        reg[i] = al.amdgpu.raw_buffer_load_x1(
            src_rsrc,
            thread_offset,
            tile_offset + i * thread_offset_stride,
            0,
        )


@avelang.jit
def _store_shm_a(
    shm: al.Tensor((SHM_TOTAL_BF16_A,), al.bf16),
    reg: al.Tensor((REG_WORDS_A,), al.u32),
    tid: al.u32,
):
    shm_words = al.view(shm, al.Tensor((SHM_TOTAL_WORDS_A,), al.u32))

    row = tid // GLOBAL_WORDS_PER_ROW
    col_word = tid - row * GLOBAL_WORDS_PER_ROW
    shm_word = row * SHM_GROUP_WORDS_A + col_word
    shm_word_stride = GLOBAL_ROWS_PER_ROUND * SHM_GROUP_WORDS_A

    for i in al.range(REG_WORDS_A):
        shm_words[shm_word + i * shm_word_stride] = reg[i]


@avelang.jit
def _store_shm_b(
    shm: al.Tensor((SHM_TOTAL_BF16_B,), al.bf16),
    reg: al.Tensor((REG_WORDS_B,), al.u32),
    tid: al.u32,
):
    shm_words = al.view(shm, al.Tensor((SHM_TOTAL_WORDS_B,), al.u32))

    row = tid // GLOBAL_WORDS_PER_ROW
    col_word = tid - row * GLOBAL_WORDS_PER_ROW
    row_group = row // SHM_READ_ROWS_B
    row_in_group = row - row_group * SHM_READ_ROWS_B
    shm_word = (
        row_group * SHM_GROUP_WORDS_B
        + row_in_group * SHM_ROW_WORDS
        + col_word
    )
    shm_word_stride = (
        (GLOBAL_ROWS_PER_ROUND // SHM_READ_ROWS_B)
        * SHM_GROUP_WORDS_B
    )

    for i in al.range(REG_WORDS_B):
        shm_words[shm_word + i * shm_word_stride] = reg[i]


@avelang.jit
def _load_shm_to_regs_batch_a(
    shm: al.Tensor((SHM_TOTAL_BF16_A,), al.bf16),
    warp_row: al.u32,
    batch_id: al.u32,
    wtid: al.u32,
    data: al.Tensor((M_TILES_PER_WARP, 2), al.u32),
):
    shm_words = al.view(shm, al.Tensor((SHM_TOTAL_WORDS_A,), al.u32))
    lane = wtid % 16
    quad = wtid // 16
    start_row = warp_row * 16 + lane
    col_uint2 = quad + batch_id * 4

    for tile in al.range(M_TILES_PER_WARP):
        uint2_index = start_row * 17 + col_uint2 + tile * 544
        word_index = uint2_index * 2
        data[tile, 0] = shm_words[word_index]
        data[tile, 1] = shm_words[word_index + 1]


@avelang.jit
def _load_shm_to_regs_batch_b(
    shm: al.Tensor((SHM_TOTAL_BF16_B,), al.bf16),
    warp_col: al.u32,
    batch_id: al.u32,
    wtid: al.u32,
    data: al.Tensor((N_TILES_PER_WARP, 2), al.u32),
):
    shm_words = al.view(shm, al.Tensor((SHM_TOTAL_WORDS_B,), al.u32))
    lane = wtid % 16
    quad = wtid // 16
    start_row = warp_col * WARP_MAT_N + lane * SHM_READ_ROWS_B
    col_uint2 = quad + batch_id * 4
    start_uint2 = (start_row // SHM_READ_ROWS_B) * (SHM_GROUP_WORDS_B // 2)

    for tile in al.range(N_TILES_PER_WARP):
        uint2_index = start_uint2 + col_uint2 + tile * 16
        word_index = uint2_index * 2
        data[tile, 0] = shm_words[word_index]
        data[tile, 1] = shm_words[word_index + 1]


@avelang.jit
def _matmul_from_regs_batch(
    data_a: al.Tensor((M_TILES_PER_WARP, 2), al.u32),
    data_b: al.Tensor((N_TILES_PER_WARP, 2), al.u32),
    acc: al.Tensor((M_TILES_PER_WARP, N_TILES_PER_WARP, 4), al.f32),
):
    for tile_m in al.range(M_TILES_PER_WARP):
        for tile_n in al.range(N_TILES_PER_WARP):
            acc[tile_m, tile_n] = al.amdgpu.mfma_16x16x16_bf16_f32(
                data_a[tile_m],
                data_b[tile_n],
                acc[tile_m, tile_n],
            )


@avelang.jit
def _write_results(
    dst_rsrc: al.Tensor((4,), al.u32),
    group_m: al.u32,
    group_n: al.u32,
    wtid: al.u32,
    warp_row: al.u32,
    warp_col: al.u32,
    acc: al.Tensor((M_TILES_PER_WARP, N_TILES_PER_WARP, 4), al.f32),
):
    lane = wtid % 16
    quad = wtid // 16
    warp_offset = (
        group_m * GROUP_M * N_SIZE
        + group_n * GROUP_N
    ) * BF16_BYTES

    for tile_m in al.range(M_TILES_PER_WARP):
        for acc_idx in al.range(4):
            row = warp_row * 16 + quad * 4 + tile_m * 32 + acc_idx
            col_base = warp_col * WARP_MAT_N + lane * SHM_READ_ROWS_B

            for t in al.range(0, N_TILES_PER_WARP, 4):
                col = col_base + t
                thread_offset = (row * N_SIZE + col) * BF16_BYTES
                lo0 = al.bitcast(acc[tile_m, t, acc_idx], al.u32)
                hi0 = al.bitcast(acc[tile_m, t + 1, acc_idx], al.u32)
                lo1 = al.bitcast(acc[tile_m, t + 2, acc_idx], al.u32)
                hi1 = al.bitcast(acc[tile_m, t + 3, acc_idx], al.u32)
                packed = al.full((2,), 0, al.u32)
                packed[0] = al.amdgpu.perm(hi0, lo0, 0x07060302)
                packed[1] = al.amdgpu.perm(hi1, lo1, 0x07060302)
                al.amdgpu.raw_buffer_store_x2(
                    packed, dst_rsrc, thread_offset, warp_offset, 0
                )


@avelang.jit
def _hot_loop_scheduler():
    for _ in al.range(45):
        al.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, 1, 0)
        al.amdgpu.sched_group_barrier(SCHED_MASK_DS_READ, 1, 0)

    al.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, 8, 0)
    al.amdgpu.sched_group_barrier(0x0800, 1, 0)
    al.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, 1, 0)

    for _ in al.range(30):
        al.amdgpu.sched_group_barrier(SCHED_MASK_DS_WRITE, 1, 0)
        al.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, 1, 0)
        al.amdgpu.sched_group_barrier(SCHED_MASK_BUFFER_LOAD, 1, 0)
        al.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, 1, 0)
        al.amdgpu.sched_group_barrier(SCHED_MASK_DS_WRITE, 1, 0)
        al.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, 1, 0)
        al.amdgpu.sched_group_barrier(SCHED_MASK_BUFFER_LOAD, 1, 0)
        al.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, 2, 0)

    al.amdgpu.sched_group_barrier(0x0800, 1, 0)

    for _ in al.range(15):
        al.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, 1, 0)
        al.amdgpu.sched_group_barrier(SCHED_MASK_DS_READ, 1, 0)

    al.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, 5, 0)


@avelang.jit
def _gemm_pipeline_transposed_b_kernel(
    A: al.Pointer(al.bf16),
    B: al.Pointer(al.bf16),
    C: al.Pointer(al.bf16),
):
    tid = al.thread_id(0)
    wid = tid // WARP_SIZE
    wtid = tid % WARP_SIZE
    warp_row = wid // WARP_PER_COL
    warp_col = wid % WARP_PER_COL

    group_m, group_n = _wgm_mapping()

    a_tensor = al.make_tensor(A, al.bf16, al.make_layout((M_SIZE, K_SIZE), (K_SIZE, 1)))
    b_tensor = al.make_tensor(B, al.bf16, al.make_layout((N_SIZE, K_SIZE), (K_SIZE, 1)))
    c_tensor = al.make_tensor(C, al.bf16, al.make_layout((M_SIZE * N_SIZE,), (1,)))
    a_rsrc = al.amdgpu.make_rsrc(a_tensor, M_SIZE * K_SIZE * BF16_BYTES)
    b_rsrc = al.amdgpu.make_rsrc(b_tensor, N_SIZE * K_SIZE * BF16_BYTES)
    c_rsrc = al.amdgpu.make_rsrc(c_tensor, M_SIZE * N_SIZE * BF16_BYTES)

    shm_a = al.make_shared((SHM_TOTAL_BF16_A,), al.bf16)
    shm_b = al.make_shared((SHM_TOTAL_BF16_B,), al.bf16)
    reg_a = al.make_local((REG_WORDS_A,), al.u32)
    reg_b = al.make_local((REG_WORDS_B,), al.u32)
    data_a0 = al.make_local((M_TILES_PER_WARP, 2), al.u32)
    data_a1 = al.make_local((M_TILES_PER_WARP, 2), al.u32)
    data_a2 = al.make_local((M_TILES_PER_WARP, 2), al.u32)
    data_a3 = al.make_local((M_TILES_PER_WARP, 2), al.u32)
    data_b0 = al.make_local((N_TILES_PER_WARP, 2), al.u32)
    data_b1 = al.make_local((N_TILES_PER_WARP, 2), al.u32)
    data_b2 = al.make_local((N_TILES_PER_WARP, 2), al.u32)
    data_b3 = al.make_local((N_TILES_PER_WARP, 2), al.u32)
    acc = al.make_local((M_TILES_PER_WARP, N_TILES_PER_WARP, 4), al.f32)

    for tile_m in al.range(M_TILES_PER_WARP):
        for tile_n in al.range(N_TILES_PER_WARP):
            for acc_idx in al.range(4):
                acc[tile_m, tile_n, acc_idx] = al.convert(0.0, al.f32)

    k_total = al.convert(K_TILES, al.u32)

    _load_global_a(a_rsrc, group_m, _staggered_k_tile(group_m, group_n, k_total, 0), tid, reg_a)
    _load_global_b(b_rsrc, group_n, _staggered_k_tile(group_m, group_n, k_total, 0), tid, reg_b)
    _store_shm_a(shm_a, reg_a, tid)
    _store_shm_b(shm_b, reg_b, tid)
    al.syncthreads()

    _load_shm_to_regs_batch_a(shm_a, warp_row, 0, wtid, data_a0)
    _load_shm_to_regs_batch_b(shm_b, warp_col, 0, wtid, data_b0)
    _load_global_a(a_rsrc, group_m, _staggered_k_tile(group_m, group_n, k_total, 1), tid, reg_a)
    _load_global_b(b_rsrc, group_n, _staggered_k_tile(group_m, group_n, k_total, 1), tid, reg_b)

    for k_idx in al.range(0, k_total - 2):
        _load_shm_to_regs_batch_a(shm_a, warp_row, 1, wtid, data_a1)
        _load_shm_to_regs_batch_b(shm_b, warp_col, 1, wtid, data_b1)
        _matmul_from_regs_batch(data_a0, data_b0, acc)

        _load_shm_to_regs_batch_a(shm_a, warp_row, 2, wtid, data_a2)
        _load_shm_to_regs_batch_b(shm_b, warp_col, 2, wtid, data_b2)
        _matmul_from_regs_batch(data_a1, data_b1, acc)

        _load_shm_to_regs_batch_a(shm_a, warp_row, 3, wtid, data_a3)
        _load_shm_to_regs_batch_b(shm_b, warp_col, 3, wtid, data_b3)
        _matmul_from_regs_batch(data_a2, data_b2, acc)

        al.syncthreads()

        _store_shm_a(shm_a, reg_a, tid)
        _store_shm_b(shm_b, reg_b, tid)
        _load_global_a(a_rsrc, group_m, _staggered_k_tile(group_m, group_n, k_total, k_idx + 2), tid, reg_a)
        _load_global_b(b_rsrc, group_n, _staggered_k_tile(group_m, group_n, k_total, k_idx + 2), tid, reg_b)
        al.syncthreads()

        _load_shm_to_regs_batch_a(shm_a, warp_row, 0, wtid, data_a0)
        _load_shm_to_regs_batch_b(shm_b, warp_col, 0, wtid, data_b0)
        _matmul_from_regs_batch(data_a3, data_b3, acc)

        _hot_loop_scheduler()


    _load_shm_to_regs_batch_a(shm_a, warp_row, 1, wtid, data_a1)
    _load_shm_to_regs_batch_b(shm_b, warp_col, 1, wtid, data_b1)
    _matmul_from_regs_batch(data_a0, data_b0, acc)

    _load_shm_to_regs_batch_a(shm_a, warp_row, 2, wtid, data_a2)
    _load_shm_to_regs_batch_b(shm_b, warp_col, 2, wtid, data_b2)
    _matmul_from_regs_batch(data_a1, data_b1, acc)

    _load_shm_to_regs_batch_a(shm_a, warp_row, 3, wtid, data_a3)
    _load_shm_to_regs_batch_b(shm_b, warp_col, 3, wtid, data_b3)
    _matmul_from_regs_batch(data_a2, data_b2, acc)

    al.syncthreads()
    _store_shm_a(shm_a, reg_a, tid)
    _store_shm_b(shm_b, reg_b, tid)
    al.syncthreads()

    _load_shm_to_regs_batch_a(shm_a, warp_row, 0, wtid, data_a0)
    _load_shm_to_regs_batch_b(shm_b, warp_col, 0, wtid, data_b0)
    _matmul_from_regs_batch(data_a3, data_b3, acc)

    _load_shm_to_regs_batch_a(shm_a, warp_row, 1, wtid, data_a1)
    _load_shm_to_regs_batch_b(shm_b, warp_col, 1, wtid, data_b1)
    _matmul_from_regs_batch(data_a0, data_b0, acc)

    _load_shm_to_regs_batch_a(shm_a, warp_row, 2, wtid, data_a2)
    _load_shm_to_regs_batch_b(shm_b, warp_col, 2, wtid, data_b2)
    _matmul_from_regs_batch(data_a1, data_b1, acc)

    _load_shm_to_regs_batch_a(shm_a, warp_row, 3, wtid, data_a3)
    _load_shm_to_regs_batch_b(shm_b, warp_col, 3, wtid, data_b3)
    _matmul_from_regs_batch(data_a2, data_b2, acc)
    _matmul_from_regs_batch(data_a3, data_b3, acc)

    _write_results(c_rsrc, group_m, group_n, wtid, warp_row, warp_col, acc)


def gemm_pipeline_transposed_b(
    A: torch.Tensor,
    B: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if A.dtype != torch.bfloat16:
        raise ValueError(f"A must have dtype torch.bfloat16, got {A.dtype}.")
    if B.dtype != torch.bfloat16:
        raise ValueError(f"B must have dtype torch.bfloat16, got {B.dtype}.")

    m = A.shape[0]
    k = A.shape[1]
    n = B.shape[0]
    if (m, n, k) != (4096, 4096, 4096):
        raise ValueError(
            "amdgpu_gemm_4096 only supports M=N=K=4096, "
            f"got M={m}, N={n}, K={k}."
        )
    if n % GROUP_N != 0:
        raise ValueError(f"N must be a multiple of {GROUP_N}, got {n}.")
    if k % GROUP_K != 0:
        raise ValueError(f"K must be a multiple of {GROUP_K}, got {k}.")

    out_shape = (m, n)
    if out is None:
        out = torch.empty(out_shape, dtype=torch.bfloat16, device=A.device)
    elif out.shape != out_shape:
        raise ValueError(f"out must have shape {out_shape}, got {tuple(out.shape)}.")
    elif out.dtype != torch.bfloat16:
        raise ValueError(f"out must have dtype torch.bfloat16, got {out.dtype}.")
    elif out.device != A.device:
        raise ValueError(f"out must be on device {A.device}, got {out.device}.")

    grid_size = GRID_SIZE
    block_size = WARP_SIZE * NUM_WARPS

    _gemm_pipeline_transposed_b_kernel[lambda: ((grid_size, 1, 1), (block_size, 1, 1))](A, B, out)
    return out


def gemm_4096_transposed_b(
    A: torch.Tensor,
    B: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    return gemm_pipeline_transposed_b(A, B, out)
