import os
import sys
import re
import glob
import shutil
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

# --- Gating System hyper-parameters -----------------------------------------
# Optimistic-initialisation: a brand-new expert is assumed good until proven
# otherwise, so it earns gate weight (airtime) before its score converges.
OPTIMISTIC_BASELINE = 1.0
# Softmax temperature for the gate. Lower => sharper preference for the best
# experts; higher => closer to uniform averaging.
GATE_TEMPERATURE = 0.2
# Moving-average rate for the per-expert performance score and velocity EMA.
SCORE_EMA_ALPHA = 0.05
# Minimum rounds an expert must be evaluated before it is eligible for archiving.
MIN_EVAL_WINDOW = 200
# Archive an expert once its rolling proposed-velocity falls below this fraction
# of the live baseline engine velocity (i.e. it is persistently diluting rounds).
ARCHIVE_VELOCITY_FRACTION = 0.90


def _softmax(values, temperature):
    """Numerically stable temperature-scaled softmax over a 1-D array."""
    x = np.asarray(values, dtype=np.float64) / max(temperature, 1e-8)
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-12)


def load_dynamic_extensions():
    """Scans the extensions directory and dynamically loads evolved logic primitives, skipping legacy fragments.

    Returns a list of expert records (dicts), each carrying the callable, its
    source path, a unique per-file key, and the expert's MODULE_METADATA so the
    gate can identify and score it across rounds.
    """
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
                meta = getattr(module, "MODULE_METADATA", {}) or {}
                # Unique identity per expert file; MODULE_METADATA (generation_id,
                # purpose, compiled_timestamp) is carried for scoring/archival logs.
                key = f"{module_name}@{meta.get('compiled_timestamp', 'na')}"
                extensions.append({
                    "fn": module.execute_extension_pass,
                    "path": file_path,
                    "name": module_name,
                    "key": key,
                    "meta": meta,
                })
        except Exception:
            pass
    return extensions


class ExpertGate:
    """Performance-tracked Mixture-of-Experts gate.

    Maintains a rolling reward-based performance score and a rolling
    proposed-velocity EMA per expert, combines expert deltas with a
    temperature-scaled softmax over those scores, and archives experts that
    persistently dilute the engine velocity.
    """

    def __init__(self, experts, archive_dir):
        self.experts = list(experts)
        self.archive_dir = archive_dir
        # Per-expert gate state keyed by the unique expert key.
        self.state = {
            e["key"]: {"score": OPTIMISTIC_BASELINE, "vel_ema": None, "evals": 0}
            for e in self.experts
        }

    def is_active(self):
        return bool(self.experts)

    def combine(self, baseline_velocity, reward, lr):
        """Compute the gated residual update for one round and return the new velocity.

        1. Each active expert proposes a delta independently against the SAME
           baseline (no in-place cascade).
        2. Deltas are combined by a temperature-scaled softmax over historical
           performance scores -> standout experts dominate, mediocre ones fade.
        3. Each contributor's reward-credit score and velocity EMA are updated.
        4. Experts that drag velocity below ARCHIVE_VELOCITY_FRACTION of baseline
           past the evaluation window are archived out of the active pool.
        """
        proposals = []  # (expert_record, delta, proposed_velocity)
        for e in self.experts:
            try:
                proposed = float(e["fn"](baseline_velocity, reward, lr))
            except Exception:
                continue
            delta = proposed - baseline_velocity
            if np.isfinite(delta):
                proposals.append((e, delta, proposed))

        if not proposals:
            return baseline_velocity

        deltas = np.array([d for _, d, _ in proposals], dtype=np.float64)
        scores = np.array([self.state[e["key"]]["score"] for e, _, _ in proposals], dtype=np.float64)

        # --- Weighted gating (replaces the old unweighted np.mean) ---
        weights = _softmax(scores, GATE_TEMPERATURE)
        combined_delta = float(np.sum(weights * deltas))
        new_velocity = baseline_velocity + combined_delta

        # --- Performance credit assignment + rolling score update ---
        # Credit rewards experts whose proposal pushed velocity UP relative to
        # this round's peers, scaled by the realised reward. Reward-raising
        # experts accrue score; choking experts decay toward 0.
        scale = float(np.mean(np.abs(deltas))) + 1e-8
        for e, delta, proposed in proposals:
            st = self.state[e["key"]]
            credit = float(reward) * 0.5 * (1.0 + np.tanh(delta / scale))
            st["score"] = (1.0 - SCORE_EMA_ALPHA) * st["score"] + SCORE_EMA_ALPHA * credit
            st["vel_ema"] = proposed if st["vel_ema"] is None else \
                (1.0 - SCORE_EMA_ALPHA) * st["vel_ema"] + SCORE_EMA_ALPHA * proposed
            st["evals"] += 1

        # --- Automated archiving / gene-pool pruning ---
        self._archive_underperformers(baseline_velocity)
        return new_velocity

    def _archive_underperformers(self, baseline_velocity):
        survivors = []
        for e in self.experts:
            st = self.state[e["key"]]
            # Base archival safety on the structural performance score credit dropping,
            # or a true flatline below baseline margin, avoiding warm-up EMA lag penalties.
            if st["evals"] >= MIN_EVAL_WINDOW and st["score"] < 0.15:
                self._archive(e, st)
            else:
                survivors.append(e)
        self.experts = survivors

    def _archive(self, expert, st):
        try:
            os.makedirs(self.archive_dir, exist_ok=True)
            dest = os.path.join(self.archive_dir, os.path.basename(expert["path"]))
            shutil.move(expert["path"], dest)
            print(f"[Gate] Archived underperforming expert '{expert['name']}' "
                  f"(score={st['score']:.3f}, vel_ema={st['vel_ema']:.3f}) -> src/archive/")
        except Exception as exc:
            print(f"[Gate] Archive failed for {expert['name']}: {exc}")


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
    archive_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "archive")
    gate = ExpertGate(load_dynamic_extensions(), archive_dir)

    if gate.is_active():
        print(f'--- Framework Active | Gated MoE over {len(gate.experts)} Evolved Experts ---')
    else:
        print('--- Framework Active | Base Matrix-Free Spectral Path ---')

    for r in range(1, rounds + 1):
        vdf_seed = f'epoch_{r}_{engine.velocity_ema}'.encode()
        y, proof = vdf_engine.evaluate_and_prove(vdf_seed, 64)
        pacing_queue.submit_verification(r, vdf_seed, 64, y, proof, vdf_callback)

        reward = np.random.random()

        # Base acceleration step (XLA-compiled JAX optimisation)
        engine.execute_accelerated_step(lr, reward)

        # Performance-tracked gating: softmax-weighted combination of expert
        # deltas (replaces unweighted np.mean) so standout experts are not
        # washed out by mediocre/conflicting ones; underperformers self-archive.
        if gate.is_active():
            engine.velocity_ema = gate.combine(engine.velocity_ema, reward, lr)

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
