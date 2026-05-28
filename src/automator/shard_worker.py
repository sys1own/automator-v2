
import os
import time
import struct
import numpy as np
from multiprocessing import Process, shared_memory

class ShardWorkerProcess(Process):
    def __init__(self, shard_id, shm_name="topo_ai_agents"):
        super().__init__()
        self.shard_id = shard_id
        self.shm_name = shm_name.strip('/').strip("/")

    def run(self):
        try:
            self.shm = shared_memory.SharedMemory(name=self.shm_name)
            while True:
                v_curr = struct.unpack_from(">d", self.shm.buf, 0)[0]
                shard_offset = (ord(self.shard_id) - ord("A")) * 8 + 8
                mutation = np.tanh(v_curr) * np.random.normal(1.0, 0.05)
                struct.pack_into(">d", self.shm.buf, shard_offset, float(mutation))
                time.sleep(0.01)
        except Exception as e:
            print(f"[Worker-{self.shard_id}] FAULT: {e}")
