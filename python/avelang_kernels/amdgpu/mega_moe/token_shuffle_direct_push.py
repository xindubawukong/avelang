"""Separate count, destination planning, and direct source-push phases."""

from functools import cache

import avelang
import avelang.language as al

from .workspace import WorkspaceLayout


@cache
def make_count(config):
    layout = WorkspaceLayout(config)
    E, LE, K = config.solution.experts, config.local_experts, config.solution.topk
    SIZE, B, SLOT, CAP = (
        layout.workspace_bytes,
        layout.rank_sym_buffer_base,
        layout.rank_slot_bytes,
        config.max_tokens_per_rank,
    )
    SEND, RECV, ROUTES = layout.send_counts, layout.recv_counts, layout.recv_tokens

    @avelang.jit
    def count_routes(heap: al.Pointer(al.u8), ids: al.Pointer(al.u32), tokens: al.u32, rank: al.u32):
        tid = al.convert(al.thread_id(0), al.u32)
        memory = al.make_tensor(heap, al.u8, al.make_layout((SIZE,), (1,)))
        resource = al.amdgpu.make_rsrc(memory, SIZE)
        routes = al.make_tensor(ids, al.u32, al.make_layout((tokens * K,), (1,)))
        counts = al.make_shared((E,), al.u32)
        for expert in al.range(tid, E, 256):
            counts[expert] = al.convert(0, al.u32)
        al.syncthreads()
        for route in al.range(tid, tokens * K, 256):
            expert = routes[route]
            if expert < E:
                ordinal = al.amdgpu.atomic_add(expert * 4, al.convert(1, al.u32), counts, 0)
                al.amdgpu.raw_buffer_store_x1(
                    al.convert(route, al.u32), resource, B + rank * SLOT + ROUTES + (expert * CAP + ordinal) * 4, 0, 17
                )
        al.syncthreads()
        for expert in al.range(tid, E, 256):
            count = counts[expert]
            al.amdgpu.raw_buffer_store_x1(count, resource, B + rank * SLOT + SEND + expert * 8, 0, 17)
            al.amdgpu.raw_buffer_store_x1(
                count, resource, B + (expert // LE) * SLOT + RECV + (rank * LE + expert % LE) * 8, 0, 17
            )
        al.amdgpu.fence(1, 2)

    return count_routes


@cache
def make_plan(config):
    layout = WorkspaceLayout(config)
    R, LE = config.solution.world_size, config.local_experts
    SIZE, C, B, SLOT = (
        layout.workspace_bytes,
        layout.barrier_record_bytes,
        layout.rank_sym_buffer_base,
        layout.rank_slot_bytes,
    )
    RECV, SUM, PLAN = layout.recv_counts, layout.recv_sum, layout.plan_base

    @avelang.jit
    def plan_routes(heap: al.Pointer(al.u8), rank: al.u32):
        tid = al.convert(al.thread_id(0), al.u32)
        memory = al.make_tensor(heap, al.u8, al.make_layout((SIZE,), (1,)))
        resource = al.amdgpu.make_rsrc(memory, SIZE)
        if tid == 0:
            pool = al.convert(0, al.u32)
            for expert in al.range(LE):
                total = al.convert(0, al.u32)
                for src in al.range(R):
                    count = al.amdgpu.raw_buffer_load_x1(
                        resource, B + rank * SLOT + RECV + (src * LE + expert) * 8, 0, 17
                    )
                    al.amdgpu.raw_buffer_store_x1(
                        pool + total, resource, src * C + PLAN + (rank * LE + expert) * 8, 0, 17
                    )
                    total = total + count
                al.amdgpu.raw_buffer_store_x1(total, resource, B + rank * SLOT + SUM + expert * 8, 0, 17)
                pool = pool + ((total + 31) // 32) * 32
            al.amdgpu.fence(1, 2)

    return plan_routes


@cache
def make_push(config):
    layout = WorkspaceLayout(config)
    LE, K = config.local_experts, config.solution.topk
    SIZE, C, B, SLOT, CAP = (
        layout.workspace_bytes,
        layout.barrier_record_bytes,
        layout.rank_sym_buffer_base,
        layout.rank_slot_bytes,
        config.max_tokens_per_rank,
    )
    ROW_BYTES = config.input_token_bytes
    SEND, ROUTES, PLAN = layout.send_counts, layout.recv_tokens, layout.plan_base
    ACT, WEIGHTS, META = layout.l1_tokens, layout.l1_weights, layout.metadata

    @avelang.jit
    def push_routes(
        heap: al.Pointer(al.u8), act: al.Pointer(al.u32), weights: al.Pointer(al.f32), tokens: al.u32, rank: al.u32
    ):
        tid, expert = al.convert(al.thread_id(0), al.u32), al.convert(al.block_id(0), al.u32)
        memory = al.make_tensor(heap, al.u8, al.make_layout((SIZE,), (1,)))
        resource = al.amdgpu.make_rsrc(memory, SIZE)
        inputs = al.make_tensor(act, al.u32, al.make_layout((tokens * ROW_BYTES // 4,), (1,)))
        source = al.amdgpu.make_rsrc(inputs, tokens * ROW_BYTES)
        route_weights = al.make_tensor(weights, al.f32, al.make_layout((tokens * K,), (1,)))
        count = al.amdgpu.raw_buffer_load_x1(resource, B + rank * SLOT + SEND + expert * 8, 0, 17)
        pool = al.amdgpu.raw_buffer_load_x1(resource, rank * C + PLAN + expert * 8, 0, 17)
        destination = expert // LE
        for ordinal in al.range(count):
            route = al.amdgpu.raw_buffer_load_x1(
                resource, B + rank * SLOT + ROUTES + (expert * CAP + ordinal) * 4, 0, 17
            )
            for vector in al.range(tid, ROW_BYTES // 16, 256):
                value = al.amdgpu.raw_buffer_load_x4(source, (route // K) * ROW_BYTES + vector * 16, 0, 0)
                al.amdgpu.raw_buffer_store_x4(
                    value, resource, B + destination * SLOT + ACT + (pool + ordinal) * ROW_BYTES + vector * 16, 0, 17
                )
            if tid == 0:
                weight = al.bitcast(route_weights[route], al.u32)
                al.amdgpu.raw_buffer_store_x1(
                    weight, resource, B + destination * SLOT + WEIGHTS + (pool + ordinal) * 4, 0, 17
                )
                metadata = al.make_local((2,), al.u32)
                metadata[0], metadata[1] = route, rank
                al.amdgpu.raw_buffer_store_x2(
                    metadata, resource, B + destination * SLOT + META + (pool + ordinal) * 8, 0, 17
                )
        al.amdgpu.fence(1, 2)

    return push_routes
