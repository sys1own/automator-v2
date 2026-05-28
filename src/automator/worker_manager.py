import os
import time
from typing import List, Dict, Tuple, Optional
from multiprocessing import Process, shared_memory
import numpy as np
from vfs.cow_overlay import MvccSnapshotRegistry
from automator.shard_worker import ShardWorkerProcess

class CacheAlignedSharedMemoryLayout:
    ALIGNMENT_OFFSET = 128
    HEAD_OFFSET = 0
    TAIL_OFFSET = 128
    PAYLOAD_OFFSET = 256

class LockFreeSharedMemoryQueue:
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
            self.buffer_view = self.shm.buf
            self._write_head(0)
            self._write_tail(0)
        else:
            self.shm = shared_memory.SharedMemory(name=self.name, create=False)
            self.buffer_view = self.shm.buf

    def _read_head(self) -> int: 
        return int.from_bytes(bytes(self.buffer_view[0:8]), 'little')

    def _write_head(self, value: int):
        self.buffer_view[0:8] = value.to_bytes(8, 'little')

    def _read_tail(self) -> int:
        return int.from_bytes(bytes(self.buffer_view[128:136]), 'little')

    def _write_tail(self, value: int):
        self.buffer_view[128:136] = value.to_bytes(8, 'little')

    def push(self, data: np.ndarray) -> bool:
        head, tail = self._read_head(), self._read_tail()
        if (head - tail) >= self.capacity: return False
        slot = head % self.capacity
        offset = 256 + (slot * self.element_size)
        view = np.ndarray(self.shape, dtype=self.dtype, buffer=self.shm.buf, offset=offset)
        view[:] = data[:]
        self._write_head(head + 1)
        return True

    def pop(self) -> Optional[np.ndarray]:
        head, tail = self._read_head(), self._read_tail()
        if tail == head: return None
        slot = tail % self.capacity
        offset = 256 + (slot * self.element_size)
        view = np.ndarray(self.shape, dtype=self.dtype, buffer=self.shm.buf, offset=offset)
        self._write_tail(tail + 1)
        return view

    def close(self): self.shm.close()
    def destroy(self): 
        self.close()
        try: self.shm.unlink()
        except: pass

class DistributedWorkerPool:
    def __init__(self, shards: List[str], registry: MvccSnapshotRegistry):
        self.shards = shards
        self.registry = registry
        self.workers = {}
        for shard_id in self.shards:
            worker = ShardWorkerProcess(shard_id=shard_id, shm_name='topo_ai_agents')
            worker.start()
            self.workers[shard_id] = worker

    def terminate_pool(self):
        for w in self.workers.values():
            if w.is_alive():
                w.terminate()
                w.join()
        self.workers.clear()