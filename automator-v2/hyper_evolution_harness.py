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

    if archetype_id == "momentum_dampener":
        module_code = f"""import jax.numpy as jnp
import numpy as np

def execute_extension_pass(v, reward, lr):
    try:
        decay = jnp.exp(-jnp.abs(v - reward) * {local_bias:.6f})
        return float(np.clip(v * decay + (reward * lr * 0.1), -25.0, 25.0))
    except Exception: return v
"""
    elif archetype_id == "laplacian_governor":
        module_code = f"""import jax.numpy as jnp
import numpy as np

def execute_extension_pass(v, reward, lr):
    try:
        stabilizer = jnp.sin(reward) * {local_threshold:.6f}
        return float(np.clip(v * jnp.abs(stabilizer), -12.0, 12.0))
    except Exception: return v
"""
    else:
        module_code = f"""import jax.numpy as jnp
import numpy as np

def execute_extension_pass(v, reward, lr):
    try:
        drift = jnp.tanh(v) * {np.random.uniform(10.0, 15.0):.4f}
        return float(np.clip(v + (drift * lr), -50.0, 50.0))
    except Exception: return v
"""

    print(f"\n[Evolving Organism] Spawning targeted extension '{file_name}'...")
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

if __name__ == '__main__':
    try:
        for _ in range(3): run_generation()
    except Exception as e:
        print(f"\n[Loop Aborted] Core variant issue: {e}")
        sys.exit(1)
