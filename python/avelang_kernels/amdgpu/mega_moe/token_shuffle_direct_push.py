"""Fixed-role planner and direct source-push transport, fused with Stage1."""

from functools import cache

import avelang
import avelang.language as al

from .synchronization import complete_stores, wait_epoch, wait_equal


@cache
def make_direct_push_token_shuffle(layout, threads, words):
    s = layout.config.solution
    R, E, LE, TOPK = s.world_size, s.experts, layout.config.local_experts, s.topk
    P = min(E, s.producer_geometry.blocks)
    CAPACITY = layout.config.max_tokens_per_rank
    C, B, SLOT = layout.barrier_record_bytes, layout.rank_sym_buffer_base, layout.rank_slot_bytes
    ROW_BYTES, VECS = layout.config.input_token_bytes, layout.config.input_token_bytes // 16
    SEND, RECV, SUM, ROUTES = layout.send_counts, layout.recv_counts, layout.recv_sum, layout.recv_tokens
    L1, WEIGHTS, META, READY = layout.l1_tokens, layout.l1_weights, layout.metadata, layout.l1_ready
    ENTRY, PLAN, DONE, PLAN_READY = layout.entry_count, layout.plan_base, layout.count_done, layout.plan_ready
    GATE, LAUNCH, HEADS, L2_READY = layout.epoch_gate, layout.launch_ready, layout.work_heads, layout.l2_ready
    WAVES, ITER = threads // 64, (E + threads - 1) // threads

    @avelang.jit
    def copy_row(
        resource: al.Tensor((4,), al.u32),
        x: al.Tensor((4,), al.u32),
        rw: al.Tensor((4,), al.u32),
        destination: al.u32,
        pool: al.u32,
        route: al.u32,
        rank: al.u32,
        vector_lane: al.u32,
        vector_stride: al.u32,
        header: al.u1,
    ):
        for vec in al.range(vector_lane, VECS, vector_stride):
            value = al.amdgpu.raw_buffer_load_x4(x, (route // TOPK) * ROW_BYTES + vec * 16, 0, 0)
            al.amdgpu.raw_buffer_store_x4(
                value, resource, B + destination * SLOT + L1 + pool * ROW_BYTES + vec * 16, 0, 17
            )
        if header:
            weight = al.amdgpu.raw_buffer_load_x1(rw, route * 4, 0, 0)
            al.amdgpu.raw_buffer_store_x1(weight, resource, B + destination * SLOT + WEIGHTS + pool * 4, 0, 17)
            metadata = al.make_local((2,), al.u32)
            metadata[0], metadata[1] = route, rank
            al.amdgpu.raw_buffer_store_x2(metadata, resource, B + destination * SLOT + META + pool * 8, 0, 17)

    @avelang.jit
    def copy_rows(
        resource: al.Tensor((4,), al.u32),
        x: al.Tensor((4,), al.u32),
        rw: al.Tensor((4,), al.u32),
        destination: al.u32,
        expert: al.u32,
        pool: al.u32,
        begin: al.u32,
        end: al.u32,
        rank: al.u32,
        tid: al.u32,
        wave: al.u32,
        lane: al.u32,
    ):
        if end - begin >= WAVES * 2:
            for ordinal in al.range(begin + wave, end, WAVES):
                route = al.convert(0, al.u32)
                if lane == 0:
                    route = al.amdgpu.raw_buffer_load_x1(
                        resource,
                        B + rank * SLOT + ROUTES + ((destination * LE + expert) * CAPACITY + ordinal) * 4,
                        0,
                        16,
                    )
                route = al.amdgpu.readfirstlane(route)
                copy_row(
                    resource, x, rw, destination, pool + ordinal, route, rank, lane, al.convert(64, al.u32), lane == 0
                )
        else:
            for ordinal in al.range(begin, end):
                route = al.amdgpu.raw_buffer_load_x1(
                    resource, B + rank * SLOT + ROUTES + ((destination * LE + expert) * CAPACITY + ordinal) * 4, 0, 16
                )
                copy_row(
                    resource,
                    x,
                    rw,
                    destination,
                    pool + ordinal,
                    route,
                    rank,
                    tid,
                    al.convert(threads, al.u32),
                    tid == 0,
                )
        complete_stores()
        al.syncthreads()
        if tid == 0:
            first, stop = pool + begin, pool + end
            for block in al.range(first // 32, (stop + 31) // 32):
                lo = al.max(first, block * 32) - block * 32
                hi = al.min(stop - block * 32, 32)
                high = al.select(hi == 32, al.convert(0xFFFFFFFF, al.u32), (1 << hi) - 1)
                low = (1 << lo) - 1
                al.amdgpu.raw_buffer_atomic_or_u32(
                    high & (low ^ 0xFFFFFFFF), resource, B + destination * SLOT + READY + block * 4, 0, 16
                )
        al.syncthreads()

    @avelang.jit
    def direct_push_token_shuffle(
        resource: al.Tensor((4,), al.u32),
        scratch: al.Tensor((words,), al.u32),
        x: al.Tensor((4,), al.u32),
        ids: al.Tensor((4,), al.u32),
        rw: al.Tensor((4,), al.u32),
        tokens: al.u32,
        rank: al.u32,
        block: al.u32,
        tid: al.u32,
    ) -> al.u32:
        wave, lane = al.amdgpu.readfirstlane(tid // 64), tid % 64
        if tid == 0:
            scratch[2 * E] = al.amdgpu.raw_buffer_atomic_add_u32(
                al.convert(1, al.u32), resource, rank * C + ENTRY + block * 4, 0, 0
            )
        al.syncthreads()
        epoch = scratch[2 * E] + 1
        parity, expected = epoch & 1, ((epoch + 1) // 2) * R
        if block == 0:
            if tid < R:
                peer = (rank + tid) % R
                al.amdgpu.raw_buffer_store_x1(epoch, resource, peer * C + LAUNCH + rank * 4, 0, 17)
                wait_epoch(resource, rank * C + LAUNCH + peer * 4, epoch)
            al.syncthreads()
            counts = al.subview(scratch, (0,), (E,), (1,))
            for i in al.static_range(ITER):
                expert = tid + i * threads
                if expert < E:
                    counts[expert] = al.convert(0, al.u32)
            al.syncthreads()
            for route in al.range(tid, tokens * TOPK, threads):
                expert = al.amdgpu.raw_buffer_load_x1(ids, route * 4, 0, 0)
                if expert < E:
                    al.amdgpu.atomic_add(expert * 4, al.convert(1, al.u32), counts, 0)
            al.syncthreads()
            for i in al.static_range(ITER):
                expert = tid + i * threads
                if expert < E:
                    count = counts[expert]
                    al.amdgpu.raw_buffer_store_x1(
                        count, resource, B + rank * SLOT + SEND + expert * 8 + parity * 4, 0, 16
                    )
                    al.amdgpu.raw_buffer_store_x1(
                        count,
                        resource,
                        B + (expert // LE) * SLOT + RECV + (rank * LE + expert % LE) * 8 + parity * 4,
                        0,
                        17,
                    )
                    counts[expert] = al.convert(0, al.u32)
            complete_stores()
            al.syncthreads()
            if tid < R:
                dest = (rank + tid) % R
                al.amdgpu.raw_buffer_store_x1(expected, resource, dest * C + DONE + (parity * R + rank) * 4, 0, 17)
            if tid < 16:
                al.amdgpu.raw_buffer_store_x1(al.convert(0, al.u32), resource, HEADS + tid * 64, 0, 16)
            if wave == 0:
                if lane < R:
                    wait_equal(resource, rank * C + DONE + (parity * R + lane) * 4, expected)
                al.amdgpu.compiler_barrier()
                total = al.convert(0, al.u32)
                if lane < LE:
                    for src in al.static_range(R):
                        count = al.amdgpu.raw_buffer_load_x1(
                            resource, B + rank * SLOT + RECV + (src * LE + lane) * 8 + parity * 4, 0, 16
                        )
                        scratch[E + src * LE + lane] = count
                        total = total + count
                padded = (total + 31) // 32 * 32
                inclusive = padded
                for shift in al.static_range(6):
                    peer = al.shuffle_up(inclusive, 1 << shift, 64)
                    inclusive = inclusive + al.select(lane >= (1 << shift), peer, al.convert(0, al.u32))
                base = inclusive - padded
                if lane < LE:
                    record = al.make_local((2,), al.u32)
                    record[0], record[1] = total, al.convert(256 * R, al.u32)
                    al.amdgpu.raw_buffer_store_x2(record, resource, B + rank * SLOT + SUM + lane * 8, 0, 16)
                    prefix = al.convert(0, al.u32)
                    for src in al.static_range(R):
                        al.amdgpu.raw_buffer_store_x1(
                            base + prefix, resource, src * C + PLAN + (rank * LE + lane) * 8 + parity * 4, 0, 17
                        )
                        prefix = prefix + scratch[E + src * LE + lane]
                pool_rows = al.shuffle(inclusive, 63, 64)
                for pool_block in al.range(lane, pool_rows // 32, 64):
                    al.amdgpu.raw_buffer_store_x1(
                        al.convert(0, al.u32), resource, B + rank * SLOT + READY + pool_block * 4, 0, 16
                    )
                    al.amdgpu.raw_buffer_store_x1(al.convert(0, al.u32), resource, L2_READY + pool_block * 4, 0, 16)
            else:
                for route in al.range((wave - 1) * 64 + lane, tokens * TOPK, threads - 64):
                    expert = al.amdgpu.raw_buffer_load_x1(ids, route * 4, 0, 0)
                    if expert < E:
                        ordinal = al.amdgpu.atomic_add(expert * 4, al.convert(1, al.u32), counts, 0)
                        al.amdgpu.raw_buffer_store_x1(
                            al.convert(route, al.u32),
                            resource,
                            B + rank * SLOT + ROUTES + (expert * CAPACITY + ordinal) * 4,
                            0,
                            16,
                        )
            complete_stores()
            al.syncthreads()
            if tid < R:
                al.amdgpu.raw_buffer_store_x1(expected, resource, tid * C + PLAN_READY + (parity * R + rank) * 4, 0, 17)
            if tid == 0:
                al.amdgpu.raw_buffer_store_x1(epoch, resource, rank * C + GATE, 0, 17)
            al.syncthreads()
        elif block <= P:
            if tid == 0:
                wait_equal(resource, rank * C + GATE, epoch)
            al.syncthreads()
            slot = block - 1
            dest = slot % R
            if tid == 0:
                wait_equal(resource, rank * C + PLAN_READY + (parity * R + dest) * 4, expected)
            al.syncthreads()
            if P == 56:
                if tid < LE:
                    scratch[tid] = al.amdgpu.raw_buffer_load_x1(
                        resource, B + rank * SLOT + SEND + (dest * LE + tid) * 8 + parity * 4, 0, 16
                    )
                    scratch[E + tid] = al.amdgpu.raw_buffer_load_x1(
                        resource, rank * C + PLAN + (dest * LE + tid) * 8 + parity * 4, 0, 16
                    )
                al.syncthreads()
                for expert in al.range(LE):
                    count, base = scratch[expert], scratch[E + expert]
                    workers = al.select(count >= 64, al.convert(P // R, al.u32), al.convert(1, al.u32))
                    worker = slot // R
                    if workers > 1 or worker == expert % (P // R):
                        part = al.select(workers == 1, al.convert(0, al.u32), worker)
                        copy_rows(
                            resource,
                            x,
                            rw,
                            dest,
                            al.convert(expert, al.u32),
                            base,
                            count * part // workers,
                            count * (part + 1) // workers,
                            rank,
                            tid,
                            wave,
                            lane,
                        )
            else:
                for task in al.range(slot, E, P):
                    expert = task // R
                    if tid == 0:
                        scratch[0] = al.amdgpu.raw_buffer_load_x1(
                            resource, B + rank * SLOT + SEND + (dest * LE + expert) * 8 + parity * 4, 0, 16
                        )
                        scratch[1] = al.amdgpu.raw_buffer_load_x1(
                            resource, rank * C + PLAN + (dest * LE + expert) * 8 + parity * 4, 0, 16
                        )
                    al.syncthreads()
                    copy_rows(
                        resource,
                        x,
                        rw,
                        dest,
                        expert,
                        scratch[1],
                        al.convert(0, al.u32),
                        scratch[0],
                        rank,
                        tid,
                        wave,
                        lane,
                    )
        if tid == 0:
            wait_equal(resource, rank * C + PLAN_READY + (parity * R + rank) * 4, expected)
        al.syncthreads()
        return epoch

    return direct_push_token_shuffle
