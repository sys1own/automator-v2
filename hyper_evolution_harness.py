import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import subprocess
import time
import re
import ast
import json
import random
import numpy as np

STATE_FILE = "context/evolution_state.json"

def load_evolution_state():
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "global_generation": 0,
        "best_velocity": 0.0,
        "mutation_history": [],
        "active_parameters": {"decay_coefficient": 0.95, "clipping_ceiling": 500.0}
    }

def save_evolution_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=4)

def stream_flight(cmd, log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    env = os.environ.copy()
    if "PYTHONPATH" not in env: env["PYTHONPATH"] = os.getcwd()

    with open(log_path, 'w') as log_file:
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env
        )
        for line in iter(process.stdout.readline, ''):
            sys.stdout.write(line)
            sys.stdout.flush()
            log_file.write(line)
            log_file.flush()
    process.stdout.close()
    return process.wait()

# --- Selection-pressure & guardian-profiler hyper-parameters ----------------
PRUNE_PEAK_FRACTION = 0.95     # prune a generation below 95% of all-time best velocity
PRUNE_ROLLING_WINDOW = 3       # rolling-average window over recent successful generations
PROFILE_ROUNDS = 50            # short profiling flight length (vs the 3000-round full channel)
PROFILE_DROP_TOLERANCE = 0.95  # candidate must reach >= 95% of the no-extension baseline velocity

def _final_velocity_from_output(stdout):
    vels = []
    for raw in re.findall(r"Velocity:\s*(\S+)", stdout or ""):
        try:
            vels.append(float(raw))
        except ValueError:
            vels.append(float("nan"))
    return (vels[-1] if vels else None), vels

def _run_profile_flight(rounds, env):
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "src.main", "--input", "tasks.json",
             "--max-rounds", str(rounds), "--learning-rate", "0.15"],
            capture_output=True, text=True, env=env, timeout=240
        )
    except subprocess.TimeoutExpired:
        return None, None, [], "TIMEOUT"
    final_v, vels = _final_velocity_from_output(proc.stdout)
    output = (proc.stdout or "") + "\n" + (proc.stderr or "")
    return proc.returncode, final_v, vels, output

def verify_extension_gate(file_path, content):
    try:
        ast.parse(content)
        compile(content, file_path, "exec")
    except Exception as e:
        print(f"[Guardian Failed] AST Syntax error: {e}. Discarding module.")
        if os.path.exists(file_path): os.remove(file_path)
        return False

    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd()

    base_rc, base_v, _, _ = _run_profile_flight(PROFILE_ROUNDS, env)
    if base_rc != 0 or base_v is None or not np.isfinite(base_v):
        print(f"[Guardian Warn] Baseline profile unavailable (rc={base_rc}, v={base_v}); screening candidate on absolute health only.")
        base_v = None

    try:
        with open(file_path, "w") as f:
            f.write(content)
    except Exception as e:
        print(f"[Guardian Failed] Could not write candidate file: {e}")
        return False

    cand_rc, cand_v, cand_vels, cand_out = _run_profile_flight(PROFILE_ROUNDS, env)

    if cand_rc != 0:
        print(f"[Guardian Failed] Candidate crashed during {PROFILE_ROUNDS}-round profile (rc={cand_rc}). Removing.")
        if os.path.exists(file_path): os.remove(file_path)
        return False

    if cand_v is None or any(not np.isfinite(v) for v in cand_vels) or "FATAL" in cand_out:
        print(f"[Guardian Failed] Candidate emitted NaN/Inf/FATAL during profile. Removing.")
        if os.path.exists(file_path): os.remove(file_path)
        return False

    if base_v is not None and cand_v < base_v * PROFILE_DROP_TOLERANCE:
        print(f"[Guardian Failed] Candidate regressed at the gate: profile velocity {cand_v:.4f} < {PROFILE_DROP_TOLERANCE:.0%} of baseline {base_v:.4f}. Rejected before full flight.")
        if os.path.exists(file_path): os.remove(file_path)
        return False

    detail = f"profile velocity {cand_v:.4f}" + (f" vs baseline {base_v:.4f}" if base_v is not None else " (no baseline)")
    print(f"[Guardian Passed] Candidate cleared {PROFILE_ROUNDS}-round profiler ({detail}): {file_path}")
    return True

NONLINEARITIES = ["tanh", "sin", "arctan", "softsign", "gauss"]
BLEND_OPS = ["mean", "geom", "max", "wsum"]
ARCHETYPE_PURPOSE = {
    "laplacian_governor": "Spectral Router Topology Controller",
    "telemetry_compactor": "Shared Memory Boundary Filter",
    "momentum_dampener": "Velocity Momentum Tracking Layer",
}

def _apply_nonlin(name, expr):
    if name == "sin":      return f"jnp.sin({expr})"
    if name == "arctan":   return f"jnp.arctan({expr})"
    if name == "softsign": return f"(({expr}) / (1.0 + jnp.abs({expr})))"
    if name == "gauss":    return f"jnp.exp(-jnp.square(jnp.clip({expr}, -3.0, 3.0)))"
    return f"jnp.tanh({expr})"

def _emit_laplacian(fv, nonlin, p):
    nl = _apply_nonlin(nonlin, f"ritz_{fv}")
    return (
        f"        lapvec_{fv} = jnp.array([v_val, -v_val, r_val, -r_val], dtype=jnp.float32)\n"
        f"        lapvec_{fv} = lapvec_{fv} / (jnp.linalg.norm(lapvec_{fv}) + 1e-8)\n"
        f"        lapstep_{fv} = lapvec_{fv} - 0.5 * (jnp.roll(lapvec_{fv}, -1) + jnp.roll(lapvec_{fv}, 1)) * {p['edge']:.4f}\n"
        f"        ritz_{fv} = jnp.dot(lapvec_{fv}, lapstep_{fv}) / (jnp.dot(lapvec_{fv}, lapvec_{fv}) + 1e-8)\n"
        f"        {fv} = jnp.clip(jnp.abs({nl}), 0.05, {p['fceil']:.3f})"
    )

def _emit_telemetry(fv, nonlin, p):
    nl = _apply_nonlin(nonlin, f"shannon_{fv} * dob_{fv}")
    return (
        f"        p1_{fv} = 1.0 / (1.0 + jnp.exp(-jnp.clip(v_jax, -10.0, 10.0)))\n"
        f"        belief_{fv} = jnp.array([p1_{fv}, 1.0 - p1_{fv}])\n"
        f"        shannon_{fv} = -jnp.sum(belief_{fv} * jnp.log(belief_{fv} + 1e-12))\n"
        f"        dob_{fv} = 1.0 - (jnp.min(belief_{fv}) / (jnp.max(belief_{fv}) + 1e-12))\n"
        f"        {fv} = jnp.clip(jnp.abs({nl}) * {p['safety']:.4f}, 0.01, {p['fceil']:.3f})"
    )

def _emit_momentum(fv, nonlin, p):
    nl = _apply_nonlin(nonlin, f"1.0 / (hessian_{fv} + 1e-8)")
    return (
        f"        trackerr_{fv} = jnp.abs(v_jax - reward_jax)\n"
        f"        regular_{fv} = jnp.maximum(1e-5, trackerr_{fv} * lr_val)\n"
        f"        hessian_{fv} = 1.0 + regular_{fv} * jnp.square(v_jax - reward_jax)\n"
        f"        {fv} = jnp.clip(jnp.abs({nl}), 0.05, {p['fceil']:.3f})"
    )

_EMITTERS = {
    "laplacian_governor": _emit_laplacian,
    "telemetry_compactor": _emit_telemetry,
    "momentum_dampener": _emit_momentum,
}

def _blend_expr(op, a, b):
    if op == "geom": return f"jnp.sqrt(jnp.abs({a} * {b}) + 1e-8)"
    if op == "max":  return f"jnp.maximum({a}, {b})"
    if op == "wsum": return f"(0.7 * {a} + 0.3 * {b})"
    return f"(0.5 * ({a} + {b}))"

def _synthesize_module_code(gen_id, velocity_delta):
    archetypes = list(_EMITTERS.keys())
    primary = random.choice(archetypes)
    blend = velocity_delta < 0.0
    secondary = random.choice([a for a in archetypes if a != primary]) if blend else None

    p = {
        "edge":      float(np.random.uniform(0.20, 1.50)),
        "safety":    float(np.random.uniform(0.50, 1.60)),
        "fceil":     float(np.random.uniform(1.50, 6.00)),
        "scale":     float(np.random.uniform(0.05, 0.50)),
        "ceil_base": float(random.choice([12.0, 25.0, 50.0])),
        "ceil_mult": float(np.random.uniform(1.50, 3.00)),
        "thresh":    float(np.random.uniform(0.50, 1.50)),
    }

    nl_primary = random.choice(NONLINEARITIES)
    blocks = [_EMITTERS[primary]("factor_a", nl_primary, p)]
    if blend:
        nl_secondary = random.choice(NONLINEARITIES)
        blocks.append(_EMITTERS[secondary]("factor_b", nl_secondary, p))
        blend_op = random.choice(BLEND_OPS)
        factor_expr = _blend_expr(blend_op, "factor_a", "factor_b")
        label = f"hybrid_{primary.split('_')[0]}_{secondary.split('_')[0]}"
        regime = f"Stochastic Hybrid ({primary} x {secondary} via {blend_op})"
        nl_meta = f"{nl_primary}+{nl_secondary}"
    else:
        factor_expr = "factor_a"
        label = primary
        regime = f"Stochastic Pure ({primary})"
        nl_meta = nl_primary
        blend_op = "none"

    reward_mod = random.choice([
        "reward_jax",
        f"({_apply_nonlin(random.choice(NONLINEARITIES), 'reward_jax')} + reward_jax)",
        f"jnp.abs({_apply_nonlin(random.choice(NONLINEARITIES), 'reward_jax')})",
    ])

    purpose_desc = ARCHETYPE_PURPOSE[primary]
    factor_code = "\n".join(blocks)

    code = f'''# ==============================================================================
# AUTONOMOUS EXTENSION LAYER: GENERATION {gen_id}
# Archetype: {purpose_desc}
# Synthesis: {regime}
# ==============================================================================
import jax.numpy as jnp
import numpy as np

MODULE_METADATA = {{
    "generation_id": {gen_id},
    "purpose": "{purpose_desc}",
    "compiled_timestamp": {time.time()},
    "hyper_parameters": {{
        "regime": "{regime}",
        "primary_archetype": "{primary}",
        "secondary_archetype": "{secondary}",
        "blend_op": "{blend_op}",
        "nonlinearity": "{nl_meta}",
        "reward_modulation": "{reward_mod}",
        "edge_weight": {p['edge']:.6f},
        "safety_ratio": {p['safety']:.6f},
        "factor_ceiling": {p['fceil']:.6f},
        "step_scale": {p['scale']:.6f},
        "threshold_bias": {p['thresh']:.6f}
    }}
}}

def verify_system_state(v, reward):
    if np.isnan(v) or np.isinf(v): return False
    if np.isnan(reward) or np.isinf(reward): return False
    return True

def execute_extension_pass(v, reward, lr):
    if not verify_system_state(v, reward): return v
    try:
        v_val, r_val, lr_val = float(v), float(reward), float(lr)
        reward_jax = jnp.array(r_val, dtype=jnp.float32)
        v_jax = jnp.array(v_val, dtype=jnp.float32)
{factor_code}
        factor = jnp.clip({factor_expr}, 0.01, {p['fceil']:.3f})
        step_delta = factor * lr_val * v_val * {reward_mod} * {p['scale']:.4f}
        ceiling = float(np.maximum({p['ceil_base']:.1f}, jnp.abs(v_val) * {p['ceil_mult']:.3f}))
        return float(np.clip(v_val + float(step_delta), 0.1, ceiling))
    except Exception: return v
'''
    return code, label

def spawn_purpose_driven_module(gen_id, velocity_delta=0.0):
    ext_dir = os.path.join("src", "extensions")
    os.makedirs(ext_dir, exist_ok=True)

    module_code, archetype_label = _synthesize_module_code(gen_id, velocity_delta)
    file_name = f"ext_{archetype_label}_gen_{gen_id}_{int(time.time())}.py"
    file_path = os.path.join(ext_dir, file_name)

    mode = "EXPLORE(blend)" if velocity_delta < 0.0 else "EXPLOIT(pure)"
    print(f"\n[Evolving Organism] Synthesising stochastic extension '{file_name}' (prev dv={velocity_delta:+.4f} -> {mode})...")
    if verify_extension_gate(file_path, module_code):
        return f"Successfully spawned purposed module extension: {file_name}"
    return "Spawned extension rejected by verification gates."

def self_refactor_engine(log_path):
    try:
        with open(log_path, 'r') as f: lines = f.readlines()
        for line in reversed(lines):
            if "Round" in line and "Velocity:" in line:
                v_match = re.search(r"Velocity: ([0-9.]+)", line)
                if v_match: return float(v_match.group(1))
    except Exception: pass
    return None

def _rolling_success_average(history, window=PRUNE_ROLLING_WINDOW):
    vels = [e.get("velocity") for e in history
            if isinstance(e.get("velocity"), (int, float))
            and str(e.get("action", "")).startswith("Successfully")]
    recent = vels[-window:]
    return (sum(recent) / len(recent)) if recent else None

def _git_sync_push(gen_id, env, max_attempts=4):
    try:
        subprocess.run(['git', 'add', 'src/extensions/', 'context/', 'repo_context_bundle.txt'], check=True, env=env)
        if subprocess.run(['git', 'diff', '--cached', '--quiet'], env=env).returncode == 0:
            print(f"[Git Sync] Generation {gen_id} produced no tracked changes; nothing to push.")
            return
        subprocess.run(['git', 'commit', '-m', f"evolution(core): generation {gen_id} structured state sync"], check=True, env=env)
    except Exception as commit_err:
        print(f"[Git Sync Warning] Could not stage/commit generation {gen_id}: {commit_err}")
        return

    local_b = subprocess.run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], capture_output=True, text=True, env=env).stdout.strip(); branch = 'main' if local_b == 'HEAD' else (local_b or 'main')

    delay = 2
    for attempt in range(1, max_attempts + 1):
        pull = subprocess.run(['git', 'pull', '--rebase', 'origin', branch], capture_output=True, text=True, env=env)
        if pull.returncode != 0:
            subprocess.run(['git', 'rebase', '--abort'], env=env)
            print(f"[Git Sync Warning] Rebase conflict on attempt {attempt}; aborted to avoid a wedged tree. Skipping push this cycle (next run will reconcile).")
            return
        push = subprocess.run(['git', 'push', 'origin', 'HEAD'], capture_output=True, text=True, env=env)
        if push.returncode == 0:
            print(f"[Git Sync] Generation {gen_id} pushed to origin/{branch} (attempt {attempt}); next evolutionary run will trigger.")
            return
        print(f"[Git Sync] Push attempt {attempt}/{max_attempts} failed (another runner may have raced); retrying in {delay}s...")
        time.sleep(delay)
        delay *= 2

    print(f"[Git Sync Warning] Could not push generation {gen_id} after {max_attempts} attempts; commit left local. Next run will reconcile and re-push.")

def run_generation():
    state = load_evolution_state()
    state["global_generation"] += 1
    gen_id = state["global_generation"]

    print(f"\n{'='*80}\n[ORGANISM GENERATION {gen_id}] EXECUTING ACTIVE FLIGHT CHANNEL\n{'='*80}")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd()

    cmd = [sys.executable, '-m', 'src.main', '--input', 'tasks.json', '--max-rounds', '3000', '--learning-rate', '0.15']
    ret = stream_flight(cmd, 'context/automator_execution.log')
    if ret != 0: raise Exception(f"Flight crashed with status code {ret}")

    current_v = self_refactor_engine('context/automator_execution.log')

    peak_floor = state["best_velocity"] * PRUNE_PEAK_FRACTION if state["best_velocity"] > 0 else None
    rolling_floor = _rolling_success_average(state["mutation_history"], PRUNE_ROLLING_WINDOW)

    prune_reasons = []
    if current_v:
        if peak_floor is not None and current_v < peak_floor:
            prune_reasons.append(f"{current_v:.4f} < {PRUNE_PEAK_FRACTION:.0%} of best {state['best_velocity']:.4f} ({peak_floor:.4f})")
        if rolling_floor is not None and current_v < rolling_floor:
            prune_reasons.append(f"{current_v:.4f} < rolling avg of last {PRUNE_ROLLING_WINDOW} successes ({rolling_floor:.4f})")

    if current_v and prune_reasons:
        print(f"\n[Natural Selection Guard] Aggressive prune | " + "; ".join(prune_reasons))
        if state["mutation_history"]:
            last_mutation = state["mutation_history"][-1]
            match = re.search(r'(ext_[\w_]+\.py)', last_mutation.get("action", ""))
            if match:
                target_file = match.group(1)
                prune_path = os.path.join("src", "extensions", target_file)
                if os.path.exists(prune_path):
                    os.remove(prune_path)
                    print(f"[Pruned] Evicted regressive code asset: {target_file}")
                last_mutation["action"] = f"PRUNED due to performance degradation (Velocity: {current_v})"

    velocity_delta = 0.0
    if current_v is not None and state["mutation_history"]:
        prev_v = state["mutation_history"][-1].get("velocity")
        if prev_v is not None:
            velocity_delta = current_v - prev_v

    action_taken = spawn_purpose_driven_module(gen_id, velocity_delta)
    print(f"[Result] {action_taken}")

    if current_v and current_v > state["best_velocity"]:
        state["best_velocity"] = current_v

    state["mutation_history"].append({
        "generation": gen_id, "timestamp": time.time(), "velocity": current_v, "action": action_taken
    })
    save_evolution_state(state)

    subprocess.run([sys.executable, '-m', 'src.main', '--input', 'tasks.json', '--max-rounds', '1'], check=True, capture_output=True, env=env)
    subprocess.run(['python', 'bundle_repo.py'], check=True, env=env)

    _git_sync_push(gen_id, env)

if __name__ == '__main__':
    print("[Evolution Engine] Initiating infinite production scaling loop...")
    generation_count = 0
    while True:
        try:
            generation_count += 1
            run_generation()
            print(f"\n[Cooldown] Generation {generation_count} complete. Sleeping 20s for fair-use safety...")
            time.sleep(20)
        except KeyboardInterrupt:
            print("\n[Loop Terminated] Infinite evolution manually stopped by operator.")
            break
        except Exception as e:
            print(f"\n[Cycle Skip] Encountered transient exception: {e}")
            print("Cooling down for 15s before attempting subsequent structural mutation...")
            time.sleep(15)
