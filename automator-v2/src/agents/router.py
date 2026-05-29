
import numpy as np
import struct
from multiprocessing import shared_memory

class SharedMemoryRing:
    def __init__(self, name='topo_ai_agents', size=1024):
        self.name = name.strip('/')
        try:
            self.shm = shared_memory.SharedMemory(name=self.name, create=True, size=size)
        except FileExistsError:
            self.shm = shared_memory.SharedMemory(name=self.name)

    def write_frame(self, data): 
        self.shm.buf[:len(data)] = data

    def close(self): 
        self.shm.close()
