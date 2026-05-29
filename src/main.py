import os
import sys
import re
import glob
import importlib.util
import json
import time
import numpy as np
import jax.numpy as jnp
from src.agents.laplacian_router import JaxMatrixFreeSpectralRouter, SparseGraphTuple
from src.automator.worker_manager import LockFreeSharedMemoryQueue
from src.automator.substrate_engine import SubstrateEngine
from src.crypto.vdf_engine import AsynchronousPacingQueue, PietrzakVDFEngine
from src.vfs.cow_overlay import MvccSnapshotRegistry

def load_dynamic_extensions():
    """Scans the extensions directory and dynamically loads evolved logic primitives, skipping legacy fragments."""
    extensions = []
    ext_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "extensions")
    if not os.path.exists(ext_dir):
        return extensions

    for file_path in glob.glob(os.path.join(ext_dir, "ext_*.py")):
        try:
            module_name = os.path.basename(file_path)[:-3]

            # DEFENSIVE GUARD: Skip absolute index entries lower than Generation 16
            gen_match = re.search(r'gen_(\d+)', module_name)
            if gen_match and int(gen_match.group(1)) < 16:
                print(f"[Loader Guard] Skipping legacy contaminant: {module_name}")
                continue

            spec = importlib.util.spec_from_file_location(f"src.extensions.{module_name}", file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if hasattr(module, "execute_extension_pass"):
                extensions.append(module.execute_extension_pass)
        except Exception:
            pass
    return extensions

def run_controller(config_path, rounds, lr):
    with open(config_path, 'r') as f: config = json.load(f)
    registry = MvccSnapshotRegistry()
    router = JaxMatrixFreeSpectralRouter(embedding_dim=4)
    vdf_engine = PietrzakVDFEngine()
    pacing_queue = AsynchronousPacingQueue(max_threads=4)

    telemetry_q = LockFreeSharedMemoryQueue(name='topo_telemetry', shape=(4,), dtype=np.float64, create=True)

    def vdf_callback(rid, status):
        if status: pass
        else: print(f'[VDF] FATAL: Verification failure at Round {rid}')

    engine = SubstrateEngine(config['agents']['shards'])
    dynamic_extensions = load_dynamic_extensions()

    if dynamic_extensions:
        print(f'--- Framework Active | Loaded {len(dynamic_extensions)} Evolved Extensions ---')
    else:
        print('--- Framework Active | Base Matrix-Free Spectral Path ---')

    for r in range(1, rounds + 1):
        vdf_seed = f'epoch_{r}_{engine.velocity_ema}'.encode()
        y, proof = vdf_engine.evaluate_and_prove(vdf_seed, 64)
        pacing_queue.submit_verification(r, vdf_seed, 64, y, proof, vdf_callback)

        reward = np.random.random()

        # Base acceleration step
        engine.execute_accelerated_step(lr, reward)

        # Execute dynamically discovered extensions autonomously
        for ext_fn in dynamic_extensions:
            try:
                engine.velocity_ema = float(ext_fn(engine.velocity_ema, reward, lr))
            except Exception:
                pass

        telemetry_q.push(np.array([engine.velocity_ema, engine.diversity_index, reward, 0.0]))

        if r == 1 or r % 500 == 0 or r == rounds:
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
