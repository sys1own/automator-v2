import os
import sys
# Programmatic self-discovery pathing: unblocks any nested shell subprocess execution paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import numpy as np
import jax.numpy as jnp
from src.agents.laplacian_router import JaxMatrixFreeSpectralRouter, SparseGraphTuple
from src.automator.worker_manager import LockFreeSharedMemoryQueue
from src.automator.substrate_engine import SubstrateEngine
from src.crypto.vdf_engine import AsynchronousPacingQueue, PietrzakVDFEngine
from src.vfs.cow_overlay import MvccSnapshotRegistry

def run_controller(config_path, rounds, lr):
    with open(config_path, 'r') as f: config = json.load(f)
    registry = MvccSnapshotRegistry()
    router = JaxMatrixFreeSpectralRouter(embedding_dim=4)
    vdf_engine = PietrzakVDFEngine()
    pacing_queue = AsynchronousPacingQueue(max_threads=4)
    
    # Zero-copy SHM Queue for high-dimensional telemetry
    telemetry_q = LockFreeSharedMemoryQueue(name='topo_telemetry', shape=(4,), dtype=np.float64, create=True)
    
    def vdf_callback(rid, status):
        if status: print(f'[VDF] Round {rid} verified asynchronously.')
        else: print(f'[VDF] FATAL: Verification failure at Round {rid}')

    engine = SubstrateEngine(config['agents']['shards'])
    print('--- Framework Orchestration Active: Matrix-Free Spectral Path ---')

    for r in range(1, rounds + 1):
        vdf_seed = f'epoch_{r}_{engine.velocity_ema}'.encode()
        # Matched both parameters to T=64 to keep cryptographic verification green and fast
        y, proof = vdf_engine.evaluate_and_prove(vdf_seed, 64)
        pacing_queue.submit_verification(r, vdf_seed, 64, y, proof, vdf_callback)

        reward = np.random.random()
        engine.execute_accelerated_step(lr, reward)
        
        telemetry_q.push(np.array([engine.velocity_ema, engine.diversity_index, reward, 0.0]))
        
        if r == 1 or r % 100 == 0 or r == rounds:
            print(f'Round {r}/{rounds} | Velocity: {engine.velocity_ema:.4f}')
        time.sleep(0.001)

    pacing_queue.shutdown()
    telemetry_q.destroy()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    args = parser.parse_args()
    run_controller(args.input, args.max_rounds, args.learning_rate)
