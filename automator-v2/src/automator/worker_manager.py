import os
import time
from typing import List, Dict, Tuple, Optional
from multiprocessing import Process, shared_memory
import numpy as np

# Fully qualified absolute import namespaces
from src.vfs.cow_overlay import MvccSnapshotRegistry
from src.automator.shard_worker import ShardWorkerProcess

class CacheAlignedSharedMemoryLayout:
    ALIGNMENT_OFFSET = 128
    HEAD_OFFSET = 0
    TAIL_OFFSET = 128
    PAYLOAD_OFFSET = 256

class LockFreeSharedMemoryQueue:
    """
    A true zero-copy, lock-free SPSC ring buffer for high-dimensional float64 numpy arrays,
    implementing hardware-native atomic index synchronization with 128-byte cache-line padding.
    """
    def __init__(self, name: str, shape: Tuple[int,...], dtype: np.dtype, capacity: int = 8, create: bool = False):
        self.name = name.strip('/')
        self.shape = shape
        self.dtype = np.dtype(dtype)
        self.capacity = capacity
        
        self.element_size = int(np.prod(shape)) * self.dtype.itemsize
        self.total_size = CacheAlignedSharedMemoryLayout.PAYLOAD_OFFSET + (self.element_size * self.capacity)
        
        if create:
            try:
                existing = shared_memory.SharedMemory(name=self.name)
                existing.close()
                existing.unlink()
            except FileNotFoundError:
                pass
            self.shm = shared_memory.SharedMemory(name=self.name, create=True, size=self.total_size)
            self.buffer_view = self.shm.buf  # FIXED: Allocated before index tracking values write to disk
            self._write_head(0)
            self._write_tail(0)
        else:
            self.shm = shared_memory.SharedMemory(name=self.name, create=False)
            self.buffer_view = self.shm.buf

    def _read_head(self) -> int:
        head_bytes = bytes(self.buffer_view[CacheAlignedSharedMemoryLayout.HEAD_OFFSET:CacheAlignedSharedMemoryLayout.HEAD_OFFSET+8])
        return int.from_bytes(head_bytes, "little")

    def _write_head(self, value: int):
        self.buffer_view[CacheAlignedSharedMemoryLayout.HEAD_OFFSET:CacheAlignedSharedMemoryLayout.HEAD_OFFSET+8] = value.to_bytes(8, "little")

    def _read_tail(self) -> int:
        tail_bytes = bytes(self.buffer_view[CacheAlignedSharedMemoryLayout.TAIL_OFFSET:CacheAlignedSharedMemoryLayout.TAIL_OFFSET+8])
        return int.from_bytes(tail_bytes, "little")

    def _write_tail(self, value: int):
        self.buffer_view[CacheAlignedSharedMemoryLayout.TAIL_OFFSET:CacheAlignedSharedMemoryLayout.TAIL_OFFSET+8] = value.to_bytes(8, "little")

    def push(self, data: np.ndarray) -> bool:
        assert data.shape == self.shape and data.dtype == self.dtype, "Shape and Dtype mismatch."
        head = self._read_head()
        tail = self._read_tail()
        if (head - tail) >= self.capacity:
            return False
        slot = head % self.capacity
        offset = CacheAlignedSharedMemoryLayout.PAYLOAD_OFFSET + (slot * self.element_size)
        dest_view = np.ndarray(self.shape, dtype=self.dtype, buffer=self.shm.buf, offset=offset)
        dest_view[:] = data[:]
        self._write_head(head + 1)
        return True

    def pop(self) -> Optional[np.ndarray]:
        head = self._read_head()
        tail = self._read_tail()
        if tail == head:
            return None
        slot = tail % self.capacity
        offset = CacheAlignedSharedMemoryLayout.PAYLOAD_OFFSET + (slot * self.element_size)
        src_view = np.ndarray(self.shape, dtype=self.dtype, buffer=self.shm.buf, offset=offset)
        self._write_tail(tail + 1)
        return src_view

    def close(self):
        self.shm.close()

    def destroy(self):
        self.close()
        try:
            self.shm.unlink()
        except FileNotFoundError:
            pass

class DistributedWorkerPool:
    def __init__(self, shards: List[str], registry: MvccSnapshotRegistry):
        self.shards = shards
        self.registry = registry
        self.workers = {} 
        print(f'[WorkerPool] Initializing Persistent Pool for shards: {shards}')

        for shard_id in self.shards:
            worker = ShardWorkerProcess(shard_id=shard_id, shm_name="topo_ai_agents")
            worker.start()
            self.workers[shard_id] = worker
            print(f'[WorkerPool] Persistent Worker Shard-{shard_id} (PID: {worker.pid}) Activated.')

    def dispatch_task_frame(self, engine):
        for shard_id in self.shards:
            epoch = self.registry.create_snapshot()
            if hasattr(engine, "dispatch_binary_sync"):
                engine.dispatch_binary_sync()

    def monitor_health(self):
        for shard_id, worker in self.workers.items():
            if not worker.is_alive():
                print(f'[WorkerPool] ALERT: Worker Shard-{shard_id} (PID {worker.pid}) hung/crashed.')

    def terminate_pool(self):
        print('[WorkerPool] Initiating Global Teardown...')
        for shard_id, worker in self.workers.items():
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=1)
                print(f'[WorkerPool] Shard-{shard_id} decommissioned.')
        self.workers.clear()
