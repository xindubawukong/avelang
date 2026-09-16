"""Intra-node HIP VMM heap with peer-owned slots at common relative offsets.

Only allocation and file-descriptor exchange run on the host. GPU kernels use
one buffer descriptor to address all ranks. Tensor views retain the allocation.
"""

import array
import ctypes as ct
import os
import socket
import sys
import uuid
from functools import cache

import torch
import torch.distributed as dist


class _Location(ct.Structure):
    _fields_ = [("type", ct.c_int), ("id", ct.c_int)]


class _AllocationProp(ct.Structure):
    _fields_ = [
        ("type", ct.c_int),
        ("handles", ct.c_int),
        ("location", _Location),
        ("metadata", ct.c_void_p),
        ("flags", ct.c_uint32),
    ]


class _Access(ct.Structure):
    _fields_ = [("location", _Location), ("flags", ct.c_int)]


@cache
def _load_hip_runtime():
    """Load HIP lazily and declare the C functions used by this allocator."""
    hip = ct.CDLL("libamdhip64.so")

    hip.hipMemAddressReserve.argtypes = [ct.POINTER(ct.c_void_p), ct.c_size_t, ct.c_size_t, ct.c_void_p, ct.c_uint64]
    hip.hipMemAddressReserve.restype = ct.c_int
    hip.hipMemAddressFree.argtypes = [ct.c_void_p, ct.c_size_t]
    hip.hipMemAddressFree.restype = ct.c_int

    hip.hipMemCreate.argtypes = [ct.POINTER(ct.c_void_p), ct.c_size_t, ct.POINTER(_AllocationProp), ct.c_uint64]
    hip.hipMemCreate.restype = ct.c_int
    hip.hipMemRelease.argtypes = [ct.c_void_p]
    hip.hipMemRelease.restype = ct.c_int

    hip.hipMemExportToShareableHandle.argtypes = [ct.POINTER(ct.c_int), ct.c_void_p, ct.c_int, ct.c_uint64]
    hip.hipMemExportToShareableHandle.restype = ct.c_int
    hip.hipMemImportFromShareableHandle.argtypes = [ct.POINTER(ct.c_void_p), ct.c_void_p, ct.c_int]
    hip.hipMemImportFromShareableHandle.restype = ct.c_int

    hip.hipMemMap.argtypes = [ct.c_void_p, ct.c_size_t, ct.c_size_t, ct.c_void_p, ct.c_uint64]
    hip.hipMemMap.restype = ct.c_int
    hip.hipMemUnmap.argtypes = [ct.c_void_p, ct.c_size_t]
    hip.hipMemUnmap.restype = ct.c_int
    hip.hipMemSetAccess.argtypes = [ct.c_void_p, ct.c_size_t, ct.POINTER(_Access), ct.c_size_t]
    hip.hipMemSetAccess.restype = ct.c_int
    hip.hipMemset.argtypes = [ct.c_void_p, ct.c_int, ct.c_size_t]
    hip.hipMemset.restype = ct.c_int

    hip.hipGetErrorString.argtypes = [ct.c_int]
    hip.hipGetErrorString.restype = ct.c_char_p
    return hip


def _check_hip(status, operation):
    if status:
        message = _load_hip_runtime().hipGetErrorString(status).decode()
        raise RuntimeError(f"{operation}: {message}")


def _exchange_fd(fd, rank, world_size, group):
    if world_size == 1:
        return {}
    address = "\0avelang-vmm-" + uuid.uuid4().hex
    descriptors = {}
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
            listener.settimeout(120)
            listener.bind(address)
            listener.listen(world_size)
            addresses = [None] * world_size
            dist.all_gather_object(addresses, address, group=group)

            def send(sock):
                sock.sendmsg([bytes([rank])], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", [fd]))])

            def receive(sock):
                data, ancillary, flags, _ = sock.recvmsg(1, socket.CMSG_SPACE(array.array("i").itemsize))
                if flags & socket.MSG_CTRUNC or len(data) != 1:
                    raise RuntimeError("truncated VMM descriptor exchange")
                for level, kind, payload in ancillary:
                    if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
                        values = array.array("i")
                        values.frombytes(payload[: values.itemsize])
                        return data[0], values[0]
                raise RuntimeError("missing VMM allocation descriptor")

            for _ in range(rank):
                connection, _ = listener.accept()
                with connection:
                    connection.settimeout(120)
                    peer, received = receive(connection)
                    descriptors[peer] = received
                    send(connection)
            for peer in range(rank + 1, world_size):
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                    connection.settimeout(120)
                    connection.connect(addresses[peer])
                    send(connection)
                    source, received = receive(connection)
                    descriptors[source] = received
        if set(descriptors) != set(range(world_size)) - {rank}:
            raise RuntimeError("invalid VMM peer ranks")
        return descriptors
    except BaseException:
        for value in descriptors.values():
            os.close(value)
        raise


class _TensorView:
    def __init__(self, owner, offset, size):
        self.owner = owner
        self.__cuda_array_interface__ = {
            "shape": (size,),
            "strides": None,
            "typestr": "|u1",
            "data": (owner.data_ptr() + offset, False),
            "version": 3,
        }


class SymmetricHeap:
    """Collectively map peer slots using the current device and process group.

    Each process must select its GPU before construction. Allocation is outside
    graph capture. Every rank must keep the heap alive until peer work finishes;
    tensor views retain local ownership automatically.
    """

    def __init__(self, layout, *, world_size, group=None):
        self.device = torch.cuda.current_device()
        self.group = group
        self.world_size = dist.get_world_size(group) if dist.is_initialized() else 1
        self.rank = dist.get_rank(group) if dist.is_initialized() else 0
        if self.world_size != world_size:
            raise ValueError("process group size does not match the requested symmetric heap size")
        self.layout = layout
        self._base = ct.c_void_p()
        self._handles, self._maps = [], []
        local_bytes = (layout.local_bytes + 4095) // 4096 * 4096
        self.size = layout.local_offset + local_bytes
        self._hip = hip = _load_hip_runtime()
        with torch.cuda.device(self.device):
            try:
                _check_hip(
                    hip.hipMemAddressReserve(ct.byref(self._base), self.size, 4096, None, 0), "hipMemAddressReserve"
                )
                self._map(0, layout.barrier_record_bytes, shared=True)
                self._map(layout.rank_sym_buffer_base, layout.rank_slot_bytes, shared=True)
                shared_end = layout.rank_sym_buffer_base + self.world_size * layout.rank_slot_bytes
                if layout.local_offset > shared_end:
                    self._map(shared_end, layout.local_offset - shared_end, shared=False)
                self._map(layout.local_offset, local_bytes, shared=False)
                access = _Access(_Location(1, self.device), 3)
                _check_hip(hip.hipMemSetAccess(self._base, self.size, ct.byref(access), 1), "hipMemSetAccess")
                for offset, size in (
                    (self.rank * layout.barrier_record_bytes, layout.barrier_record_bytes),
                    (layout.rank_base(self.rank), layout.rank_slot_bytes),
                    (layout.local_offset, local_bytes),
                ):
                    _check_hip(hip.hipMemset(self.data_ptr() + offset, 0, size), "hipMemset")
                if layout.local_offset > shared_end:
                    _check_hip(
                        hip.hipMemset(self.data_ptr() + shared_end, 0, layout.local_offset - shared_end), "hipMemset"
                    )
                torch.cuda.synchronize(self.device)
                if self.world_size > 1:
                    dist.barrier(group=group)
            except BaseException:
                self._release()
                raise

    def _map(self, offset, size, *, shared):
        hip = self._hip
        prop = _AllocationProp(1, 1 if shared else 0, _Location(1, self.device), None, 0)
        handle = ct.c_void_p()
        _check_hip(hip.hipMemCreate(ct.byref(handle), size, ct.byref(prop), 0), "hipMemCreate")
        self._handles.append(handle)
        descriptors = {}
        try:
            if shared:
                fd = ct.c_int(-1)
                _check_hip(
                    hip.hipMemExportToShareableHandle(ct.byref(fd), handle, 1, 0), "hipMemExportToShareableHandle"
                )
                try:
                    descriptors = _exchange_fd(fd.value, self.rank, self.world_size, self.group)
                finally:
                    os.close(fd.value)
            for peer in range(self.world_size if shared else 1):
                allocation = handle
                if shared and peer != self.rank:
                    allocation = ct.c_void_p()
                    _check_hip(
                        hip.hipMemImportFromShareableHandle(ct.byref(allocation), descriptors[peer], 1),
                        "hipMemImportFromShareableHandle",
                    )
                    self._handles.append(allocation)
                address = self.data_ptr() + offset + peer * size
                _check_hip(hip.hipMemMap(address, size, 0, allocation, 0), "hipMemMap")
                self._maps.append((address, size))
        finally:
            for value in descriptors.values():
                os.close(value)

    def data_ptr(self):
        return self._base.value

    def tensor(self, offset=0, size=None):
        size = self.size - offset if size is None else size
        if not 0 <= offset <= self.size or not 0 <= size <= self.size - offset:
            raise ValueError("tensor view exceeds symmetric heap")
        return torch.as_tensor(_TensorView(self, offset, size), device=f"cuda:{self.device}")

    def _release(self):
        if not self._base.value:
            return
        hip = self._hip
        with torch.cuda.device(self.device):
            for address, size in reversed(self._maps):
                hip.hipMemUnmap(address, size)
            for handle in reversed(self._handles):
                hip.hipMemRelease(handle)
            hip.hipMemAddressFree(self._base, self.size)
        self._base = ct.c_void_p()
        self._maps.clear()
        self._handles.clear()

    def __del__(self):
        if sys is not None and not sys.is_finalizing() and getattr(self, "_base", None) and self._base.value:
            self._release()
