
import os
import time
import subprocess
import struct
from typing import List, Dict
from multiprocessing import Process
from vfs.cow_overlay import MvccSnapshotRegistry
from automator.shard_worker import ShardWorkerProcess

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
