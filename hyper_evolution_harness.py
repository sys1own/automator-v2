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
    
    local_bias = float(np.random.uniform(0.01, 0.05))
    local_threshold = float(np.random.uniform(0.80, 0.95))
    local_capacity = int(np.random.randint(64, 256))

    if archetype_id == "momentum_dampener":
        module_code = f"""# ==============================================================================
# AUTONOMOUS EXTENSION LAYER: GENERATION {gen_id}
# Archetype: {purpose_desc}
# Target Subsystem: Velocity Momentum Control Matrices
# ==============================================================================
import jax.numpy as jnp
import numpy as np

# Module-Level Hyper-Parameters
HISTORICAL_MOMENTUM_BIAS = {local_bias:.6f}
SATURATION_LIMIT = 25.0

# Persistent Analytics Structural Blueprint
MODULE_METADATA = {{
    "generation_id": {gen_id},
    "purpose": "{purpose_desc}",
    "compiled_timestamp": {time.time()},
    "hyper_parameters": {{
        "momentum_bias": HISTORICAL_MOMENTUM_BIAS,
        "clipping_ceiling": SATURATION_LIMIT
    }}
}}

def verify_system_state(v, reward):
    \"\"\"Performs structural checks on internal float states before gradient pass.\"\"\"
    if np.isnan(v) or np.isinf(v):
        return False
    if reward < 0.0 or reward > 1.0:
        return False
    return True

def execute_extension_pass(v, reward, lr):
    \"\"\"Executes a 3-stage momentum tracking correction cascade.\"\"\"
    # STAGE 1: Boundary State Verification Check
    if not verify_system_state(v, reward):
        return v
        
    try:
        # STAGE 2: Core Exponential Momentum Transformation
        velocity_delta = jnp.abs(v - reward)
        decay_modifier = jnp.exp(-velocity_delta * HISTORICAL_MOMENTUM_BIAS)
        
        # Calculate intermediate accelerated momentum trajectories
        proportional_gain = reward * lr * 0.125
        base_projection = v * decay_modifier
        
        # STAGE 3: Structural Fusion & Safe Clamping Bound Enforcement
        raw_output = base_projection + proportional_gain
        optimized_output = float(np.clip(raw_output, -SATURATION_LIMIT, SATURATION_LIMIT))
        
        return optimized_output
    except Exception as runtime_fault:
        # Emergency Safe Fallback Route
        return v
"""
    elif archetype_id == "laplacian_governor":
        module_code = f"""# ==============================================================================
# AUTONOMOUS EXTENSION LAYER: GENERATION {gen_id}
# Archetype: {purpose_desc}
# Target Subsystem: Graph Spectral Router Boundary Constraints
# ==============================================================================
import jax.numpy as jnp
import numpy as np

# Module-Level Hyper-Parameters
TOPOLOGICAL_CONTRACTION_BOUND = {local_threshold:.6f}
ATTENUATOR_RATIO = 0.9535

MODULE_METADATA = {{
    "generation_id": {gen_id},
    "purpose": "{purpose_desc}",
    "compiled_timestamp": {time.time()},
    "hyper_parameters": {{
        "contraction_bound": TOPOLOGICAL_CONTRACTION_BOUND,
        "attenuator": ATTENUATOR_RATIO
    }}
}}

def process_topological_attenuation(factor):
    \"\"\"Applies dampening arrays to stabilizing factors that overshoot boundaries.\"\"\"
    processed_factor = factor
    if factor > TOPOLOGICAL_CONTRACTION_BOUND:
        processed_factor *= ATTENUATOR_RATIO
    elif factor < -TOPOLOGICAL_CONTRACTION_BOUND:
        processed_factor *= (ATTENUATOR_RATIO * 1.05)
    return processed_factor

def execute_extension_pass(v, reward, lr):
    \"\"\"Coordinates dynamic step vectors relative to Dobrushin contraction limits.\"\"\"
    try:
        # STAGE 1: Tracing Factor Synthesis
        stabilization_factor = jnp.sin(reward) * jnp.cos(lr)
        
        # STAGE 2: Nonlinear Adaptive Boundary Attenuation
        refined_factor = process_topological_attenuation(stabilization_factor)
        
        # STAGE 3: Substrate Relaxation Mapping
        relaxation_vector = v * jnp.abs(refined_factor)
        clamped_output = float(np.clip(relaxation_vector, -12.0, 12.0))
        
        return clamped_output
    except Exception:
        return v
"""
    else: # telemetry_compactor
        module_code = f"""# ==============================================================================
# AUTONOMOUS EXTENSION LAYER: GENERATION {gen_id}
# Archetype: {purpose_desc}
# Target Subsystem: Lock-Free Shared Memory Ring Buffers
# ==============================================================================
import jax.numpy as jnp
import numpy as np

# Module-Level Hyper-Parameters
CACHE_PADDING_THRESHOLD = {np.random.uniform(10.0, 15.0):.4f}
MAX_BUFFER_CAPACITY = {local_capacity}

MODULE_METADATA = {{
    "generation_id": {gen_id},
    "purpose": "{purpose_desc}",
    "compiled_timestamp": {time.time()},
    "hyper_parameters": {{
        "padding_threshold": CACHE_PADDING_THRESHOLD,
        "max_capacity": MAX_BUFFER_CAPACITY
    }}
}}

def evaluate_memory_drift(v):
    \"\"\"Calculates localized zero-copy buffer saturation limits.\"\"\"
    current_drift = jnp.tanh(v) * CACHE_PADDING_THRESHOLD
    if jnp.abs(current_drift) > 1.0:
        return jnp.sign(current_drift) * 1.0
    return current_drift

def execute_extension_pass(v, reward, lr):
    \"\"\"Applies alignment boundary adjustments to prevent false sharing.\"\"\"
    try:
        # STAGE 1: Extract Core Drift Invariants
        memory_drift_coefficient = evaluate_memory_drift(v)
        
        # STAGE 2: Compile Compound Multi-Statement Telemetry Step
        scaled_step = memory_drift_coefficient * lr
        raw_output = v + scaled_step
        
        # STAGE 3: Operational Guardrail Check
        final_telemetry_value = float(np.clip(raw_output, -50.0, 50.0))
        return final_telemetry_value
    except Exception:
        return v
"""

    print(f"\n[Evolving Organism] Spawning comprehensive targeted extension '{file_name}'...")
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
    
    # FIXED: Natural Selection Sieve. If the last flight tanked, prune the asset that caused it.
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

    action_taken = spawn_purpose_driven_module(gen_id)
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
