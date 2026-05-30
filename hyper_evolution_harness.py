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
        "active_parameters": {"decay_coefficient": 0.95, "clipping_ceiling": 12.0}
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

def verify_extension_gate(file_path, content):
    try:
        ast.parse(content)
        compile(content, file_path, "exec")

        with open(file_path, "w") as f:
            f.write(content)

        env = os.environ.copy()
        env["PYTHONPATH"] = os.getcwd()

        test_run = subprocess.run(
            [sys.executable, "-m", "src.main", "--input", "tasks.json", "--max-rounds", "2"],
            capture_output=True, text=True, env=env
        )

        if test_run.returncode == 0:
            print(f"[Guardian Passed] Spawned extension validated cleanly: {file_path}")
            return True
        else:
            print(f"[Guardian Failed] Runtime error in spawned logic. Removing extension.")
            if os.path.exists(file_path): os.remove(file_path)
            return False
    except Exception as e:
        print(f"[Guardian Failed] AST Syntax error: {e}. Discarding module.")
        if os.path.exists(file_path): os.remove(file_path)
        return False

# ---------------------------------------------------------------------------
# Structural stochastic mutation engine
#
# Instead of a rigid modulo cycle over three fixed template strings, the code
# generator now composes expert modules from vetted, always-valid building
# blocks with three layers of stochastic variation:
#   1. Algorithmic math variation  -> random nonlinearity primitive injected
#      into the factor computation and into how reward modulates the step.
#   2. Dynamic parameter synthesis  -> a negative previous-generation velocity
#      delta triggers EXPLORE mode (blend two distinct archetypes); a positive
#      delta triggers EXPLOIT mode (refine a single pure archetype).
#   3. Stochastic meta-hyperparameters -> broadened random ranges baked into
#      MODULE_METADATA so generated code explores deeper scaling regimes.
# Every composition keeps reward as a bounded scale amplifier (never a
# target-seeking subtraction) and is clip-bounded, so it always compiles and
# runs through the unchanged verify_extension_gate.
# ---------------------------------------------------------------------------
NONLINEARITIES = ["tanh", "sin", "arctan", "softsign", "gauss"]
BLEND_OPS = ["mean", "geom", "max", "wsum"]
ARCHETYPE_PURPOSE = {
    "laplacian_governor": "Spectral Router Topology Controller",
    "telemetry_compactor": "Shared Memory Boundary Filter",
    "momentum_dampener": "Velocity Momentum Tracking Layer",
}

def _apply_nonlin(name, expr):
    """Return a bounded JAX nonlinearity applied to `expr` (all R -> bounded)."""
    if name == "sin":      return f"jnp.sin({expr})"
    if name == "arctan":   return f"jnp.arctan({expr})"
    if name == "softsign": return f"(({expr}) / (1.0 + jnp.abs({expr})))"
    if name == "gauss":    return f"jnp.exp(-jnp.square(jnp.clip({expr}, -3.0, 3.0)))"
    return f"jnp.tanh({expr})"  # default / "tanh"

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
    return f"(0.5 * ({a} + {b}))"  # default / "mean"

def _synthesize_module_code(gen_id, velocity_delta):
    """Compose a stochastic, structurally-mutated expert module as source text.

    Returns (code_str, archetype_label).
    """
    archetypes = list(_EMITTERS.keys())
    primary = random.choice(archetypes)

    # Dynamic parameter synthesis: regress (dv<0) -> EXPLORE by blending two
    # distinct archetypes; improve/flat (dv>=0) -> EXPLOIT a single archetype.
    blend = velocity_delta < 0.0
    secondary = random.choice([a for a in archetypes if a != primary]) if blend else None

    # Stochastic meta-hyperparameters (broadened ranges for deeper regimes).
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

    # Algorithmic math variation on how reward modulates the (healthy) step.
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
        # STAGE 1: State vector setup (reward retained as a scale amplifier)
        v_val, r_val, lr_val = float(v), float(reward), float(lr)
        reward_jax = jnp.array(r_val, dtype=jnp.float32)
        v_jax = jnp.array(v_val, dtype=jnp.float32)

        # STAGE 2: Stochastically-synthesised factor computation
{factor_code}
        factor = jnp.clip({factor_expr}, 0.01, {p['fceil']:.3f})

        # STAGE 3: Relative step scaling (reward as directional variance amplifier)
        step_delta = factor * lr_val * v_val * {reward_mod} * {p['scale']:.4f}
        ceiling = float(np.maximum({p['ceil_base']:.1f}, jnp.abs(v_val) * {p['ceil_mult']:.3f}))
        return float(np.clip(v_val + float(step_delta), 0.1, ceiling))
    except Exception: return v
'''
    return code, label

def spawn_purpose_driven_module(gen_id, velocity_delta=0.0):
    """Synthesise a stochastic, structurally-mutated expert module and validate it through the unchanged AST/runtime gate."""
    ext_dir = os.path.join("src", "extensions")
    os.makedirs(ext_dir, exist_ok=True)

    module_code, archetype_label = _synthesize_module_code(gen_id, velocity_delta)
    file_name = f"ext_{archetype_label}_gen_{gen_id}_{int(time.time())}.py"
    file_path = os.path.join(ext_dir, file_name)

    mode = "EXPLORE(blend)" if velocity_delta < 0.0 else "EXPLOIT(pure)"
    print(f"\n[Evolving Organism] Synthesising stochastic extension '{file_name}' "
          f"(prev dv={velocity_delta:+.4f} -> {mode})...")
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

    if current_v and state["best_velocity"] > 0 and current_v < (state["best_velocity"] * 0.40):
        print(f"\n[Natural Selection Guard] Performance degraded to {current_v}. Pruning underperforming predecessor...")
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

    # Net velocity delta of the previous generation drives explore/exploit synthesis.
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

    try:
        subprocess.run(['git', 'add', 'src/extensions/', 'context/', 'repo_context_bundle.txt'], check=True, env=env)
        subprocess.run(['git', 'commit', '-m', f"evolution(core): generation {gen_id} structured state sync"], check=True, env=env)
        subprocess.run(['git', 'push', 'origin', 'main'], check=True, env=env)
        print(f"[Git Sync] Generation {gen_id} extensions cleanly pushed to remote origin.")
    except Exception as git_err:
        print(f"[Git Sync Warning] Skipping automatic repository upstream push: {git_err}")

if __name__ == '__main__':
    try:
        for _ in range(3): run_generation()
    except Exception as e:
        print(f"\n[Loop Aborted] Core variant issue: {e}")
        sys.exit(1)
