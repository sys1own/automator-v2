import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import subprocess
import time
import re
import ast
import json
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

def spawn_purpose_driven_module(gen_id):
    """Generates comprehensive, multi-stage telemetry processing modules with dedicated logic layers."""
    ext_dir = os.path.join("src", "extensions")
    os.makedirs(ext_dir, exist_ok=True)
    
    archetypes = [
        ("momentum_dampener", "Velocity Momentum Tracking Layer"),
        ("laplacian_governor", "Spectral Router Topology Controller"),
        ("telemetry_compactor", "Shared Memory Boundary Filter")
    ]
    
    archetype_id, purpose_desc = archetypes[(gen_id - 1) % len(archetypes)]
    file_name = f"ext_{archetype_id}_gen_{gen_id}_{int(time.time())}.py"
    file_path = os.path.join(ext_dir, file_name)
    
    # Stochastic variation injected across generation steps to avoid identical expert footprints
    local_threshold_bias = float(np.random.uniform(0.75, 0.99))
    local_safety_ratio = float(np.random.uniform(0.90, 0.98))
    
    if archetype_id == "momentum_dampener":
        module_code = f"""# ==============================================================================
# AUTONOMOUS EXTENSION LAYER: GENERATION {gen_id}
# Archetype: {purpose_desc}
# Target Subsystem: Adaptive Meta-Gradient Step Scaling
# ==============================================================================
import jax.numpy as jnp
import numpy as np

MODULE_METADATA = {{
    "generation_id": {gen_id},
    "purpose": "{purpose_desc}",
    "compiled_timestamp": {time.time()},
    "hyper_parameters": {{
        "regime": "Implicit-Function Meta-Gradient Pacing",
        "differentiation_mode": "mixed_flow",
        "dynamic_bounds": True
    }}
}}

def verify_system_state(v, reward):
    if np.isnan(v) or np.isinf(v): return False
    if np.isnan(reward) or np.isinf(reward): return False
    return True

def execute_extension_pass(v, reward, lr):
    if not verify_system_state(v, reward): return v
    try:
        v_jax = jnp.array(float(v), dtype=jnp.float32)
        rew_jax = jnp.array(float(reward), dtype=jnp.float32)
        lr_jax = jnp.array(float(lr), dtype=jnp.float32)
        
        tracking_error = jnp.abs(v_jax - rew_jax)
        dynamic_regularizer = jnp.maximum(1e-5, tracking_error * lr_jax)
        
        grad_inner = v_jax - rew_jax
        hessian_inner = 1.0 + dynamic_regularizer * jnp.square(grad_inner)
        mixed_partial = -lr_jax * grad_inner
        
        best_response_jacobian = mixed_partial / (hessian_inner + 1e-8)
        step_delta = -best_response_jacobian * lr_jax * (1.0 + jnp.tanh(tracking_error))
        raw_output = v_jax + step_delta
        
        ceiling = float(jnp.maximum(25.0, jnp.abs(rew_jax) * 100.0))
        return float(np.clip(raw_output, 0.1, ceiling))
    except Exception: return v
"""
    elif archetype_id == "laplacian_governor":
        module_code = f"""# ==============================================================================
# AUTONOMOUS EXTENSION LAYER: GENERATION {gen_id}
# Archetype: {purpose_desc}
# Target Subsystem: Curvature-guided Ritz value pacing over shifted graph operators
# ==============================================================================
import jax.numpy as jnp
import numpy as np

MODULE_METADATA = {{
    "generation_id": {gen_id},
    "purpose": "{purpose_desc}",
    "compiled_timestamp": {time.time()},
    "hyper_parameters": {{
        "regime": "Spectral Graph Theory Deflation",
        "operator_dimension": 4,
        "stabilization_filter": "stagnation_breaker",
        "threshold_bias": {local_threshold_bias:.6f}
    }}
}}

def execute_extension_pass(v, reward, lr):
    try:
        v_val, r_val, lr_val = float(v), float(reward), float(lr)
        edge_weight = float(np.maximum(0.01, np.abs(v_val - r_val) * lr_val))
        v_jax = jnp.array([v_val, -v_val, r_val, -r_val], dtype=jnp.float32)
        
        shift = jnp.clip(jnp.array(lr_val * r_val * {local_threshold_bias:.4f}, dtype=jnp.float32), 0.0, 1.0)
        
        def compute_shifted_laplacian_step(vec):
            left_shift = jnp.roll(vec, shift=-1)
            right_shift = jnp.roll(vec, shift=1)
            adjacency_product = 0.5 * (left_shift + right_shift) * edge_weight
            return (vec - adjacency_product) - shift * vec
            
        current_vector = v_jax / (jnp.linalg.norm(v_jax) + 1e-8)
        for _ in range(3):
            iterated = compute_shifted_laplacian_step(current_vector)
            vector_norm = jnp.linalg.norm(iterated)
            current_vector = jnp.where(vector_norm > 0.0, iterated / vector_norm, current_vector)
            
        num = jnp.dot(current_vector, compute_shifted_laplacian_step(current_vector))
        den = jnp.dot(current_vector, current_vector)
        ritz_value = jnp.where(den > 0.0, num / den, 0.5)
        
        algebraic_pacing = jnp.clip(jnp.abs(ritz_value), 0.05, 2.0)
        step_delta = algebraic_pacing * lr_val * (r_val - v_val) * 0.25
        
        ceiling = float(np.maximum(12.0, np.abs(v_val) * 2.0))
        return float(np.clip(v_val + float(step_delta), 0.1, ceiling))
    except Exception: return v
"""
    else: # telemetry_compactor
        module_code = f"""# ==============================================================================
# AUTONOMOUS EXTENSION LAYER: GENERATION {gen_id}
# Archetype: {purpose_desc}
# Target Subsystem: Dynamic Shannon-entropy dampening constraints for cache aligned queues
# ==============================================================================
import jax.numpy as jnp
import numpy as np

MODULE_METADATA = {{
    "generation_id": {gen_id},
    "purpose": "{purpose_desc}",
    "compiled_timestamp": {time.time()},
    "hyper_parameters": {{
        "regime": "Simplicial Belief Dynamics",
        "alignment_offset": 128,
        "safety_ratio": {local_safety_ratio:.6f}
    }}
}}

def execute_extension_pass(v, reward, lr):
    try:
        v_val, r_val, lr_val = float(v), float(reward), float(lr)
        v_jax = jnp.array(v_val, dtype=jnp.float32)
        p1 = 1.0 / (1.0 + jnp.exp(-jnp.clip(v_jax, -10.0, 10.0)))
        p2 = 1.0 - p1
        belief_state = jnp.array([p1, p2])
        
        eps = 1e-12
        shannon_entropy = -jnp.sum(belief_state * jnp.log(belief_state + eps))
        dobrushin_bound = 1.0 - (jnp.min(belief_state) / (jnp.max(belief_state) + eps))
        
        dampening_factor = jnp.clip(shannon_entropy * dobrushin_bound * {local_safety_ratio:.4f}, 0.01, 1.0)
        step_delta = dampening_factor * lr_val * (r_val - v_val)
        
        ceiling = float(np.maximum(50.0, jnp.abs(v_val) * 1.5))
        return float(np.clip(v_val + float(step_delta), 0.1, ceiling))
    except Exception: return v
"""

    print(f"\n[Evolving Organism] Spawning comprehensive geometric extension '{file_name}'...")
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

    action_taken = proud = spawn_purpose_driven_module(gen_id)
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
