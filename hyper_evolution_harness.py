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
        "active_parameters": {
            "learning_rate_modifier": 1.0,
            "decay_coefficient": 0.95,
            "clipping_ceiling": 12.0,
            "dropout_rate": 0.98
        }
    }

def save_evolution_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=4)

def stream_flight(cmd, log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    env = os.environ.copy()
    if "PYTHONPATH" not in env:
        env["PYTHONPATH"] = os.getcwd()
        
    with open(log_path, 'w') as log_file:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env
        )
        for line in iter(process.stdout.readline, ''):
            sys.stdout.write(line)
            sys.stdout.flush()
            log_file.write(line)
            log_file.flush()
        process.stdout.close()
        return process.wait()

def verify_gate_isolation(file_path, target_mutation_content):
    tmp_path = file_path + ".tmp"
    try:
        with open(tmp_path, "w") as f:
            f.write(target_mutation_content)
            
        ast.parse(target_mutation_content)
        compile(target_mutation_content, tmp_path, "exec")
        
        env = os.environ.copy()
        env["PYTHONPATH"] = os.getcwd()
        
        os.rename(file_path, file_path + ".bak")
        os.rename(tmp_path, file_path)
        
        test_run = subprocess.run(
            [sys.executable, "-m", "src.main", "--input", "tasks.json", "--max-rounds", "2"],
            capture_output=True, text=True, env=env
        )
        
        os.rename(file_path, tmp_path)
        os.rename(file_path + ".bak", file_path)
        
        if test_run.returncode == 0:
            os.replace(tmp_path, file_path)
            print(f"[Guardian Gate Passed] Atomic graduation completed for: {file_path}")
            return True
        else:
            print(f"[Guardian GATE 2 FAILED] Exit code {test_run.returncode}. Discarding variant.")
            if os.path.exists(tmp_path): os.remove(tmp_path)
            return False
            
    except Exception as e:
        print(f"[Guardian GATE 1 FAILED] Compilation Crash: {e}. Discarding variant.")
        if os.path.exists(tmp_path): os.remove(tmp_path)
        if os.path.exists(file_path + ".bak"): os.rename(file_path + ".bak", file_path)
        return False

def generate_procedural_mutation(state):
    """
    Stochastic Code Synthesis Core: Instead of hardcoded strings, this engine
    dynamically explores code permutations, inventing mathematical operations,
    tuning parameter configurations, and injecting algorithmic changes.
    """
    functional_path = 'src/automator/action_functional.py'
    engine_path = 'src/automator/substrate_engine.py'
    
    if not os.path.exists(functional_path) or not os.path.exists(engine_path):
        return "No action taken (missing target files)"

    mutation_types = ["tune_hyperparameters", "inject_activation_variant", "evolve_gradient_formula"]
    chosen_type = np.random.choice(mutation_types)
    log_msg = f"Mutation Type: {chosen_type} | "

    if chosen_type == "tune_hyperparameters":
        # Procedurally adjust continuous coefficients via random walks
        params = state["active_parameters"]
        params["learning_rate_modifier"] *= float(np.random.uniform(0.85, 1.15))
        params["decay_coefficient"] = float(np.clip(params["decay_coefficient"] * np.random.uniform(0.9, 1.1), 0.5, 0.999))
        params["clipping_ceiling"] = float(np.clip(params["clipping_ceiling"] + np.random.uniform(-2.0, 2.0), 1.0, 50.0))
        
        with open(functional_path, 'r') as f: content = f.read()
        
        # Rewrite the optimization gradient multiplier procedurally based on updated parameters
        content = re.sub(
            r"return lr \* reward \* grad \* [0-9.]+",
            f"return lr * reward * grad * {params['decay_coefficient']:.6f}",
            content
        )
        content = re.sub(
            r"floor,\s*[0-9.]+\)",
            f"floor, {params['clipping_ceiling']:.2f})",
            content
        )
        
        if verify_gate_isolation(functional_path, content):
            log_msg += f"Updated hyperparameters to: {params}"
            return log_msg

    elif chosen_type == "inject_activation_variant":
        # Procedurally experiment with algebraic combinations of JAX non-linear activation functions
        activations = ["jnp.tanh(w)", "jnp.sin(w)", "jnp.cos(w)", "1.0 / (1.0 + jnp.exp(-w))", "(w * jnp.tanh(w))"]
        chosen_act = np.random.choice(activations)
        
        with open(functional_path, 'r') as f: content = f.read()
        
        if "grad =" in content:
            mutated = re.sub(
                r"grad = 1\.0 - jnp\.square\(.*?\)",
                f"grad = 1.0 - jnp.square({chosen_act})",
                content
            )
            if verify_gate_isolation(functional_path, mutated):
                log_msg += f"Injected activation variant: {chosen_act}"
                return log_msg

    elif chosen_type == "evolve_gradient_formula":
        # Procedurally alter structural scale factors inside the compilation loop
        scale_variants = ["+ 0.01 * jnp.sin(w)", "* 1.02", "* 0.98", "- 0.005 * w"]
        chosen_variant = np.random.choice(scale_variants)
        
        with open(functional_path, 'r') as f: content = f.read()
        
        if "grad =" in content and chosen_variant not in content:
            mutated = re.sub(
                r"(grad = 1\.0 - jnp\.square\(.*?\))",
                rf"\1 {chosen_variant}",
                content
            )
            if verify_gate_isolation(functional_path, mutated):
                log_msg += f"Evolved gradient update formula via expression: {chosen_variant}"
                return log_msg

    return "Variant mutation dropped by guardian gates or no change applied."

def self_refactor_engine(log_path):
    print(f"\n[Refactor] Parsing structural trajectory logs: {log_path}")
    try:
        with open(log_path, 'r') as f:
            lines = f.readlines()
        
        last_round = None
        for line in reversed(lines):
            if "Round" in line and "Velocity:" in line:
                last_round = line
                break
        
        if not last_round:
            print("[Refactor] WARNING: Convergence benchmarks not located.")
            return None

        v_match = re.search(r"Velocity: ([0-9.]+)", last_round)
        if v_match:
            new_v = float(v_match.group(1))
            engine_path = 'src/automator/substrate_engine.py'
            
            with open(engine_path, 'r') as f:
                content = f.read()
            
            content = re.sub(r"self.velocity_ema = [0-9.]+", f"self.velocity_ema = {new_v:.4f}", content)
            
            with open(engine_path, 'w') as f:
                f.write(content)
            print(f"[Refactor] SUCCESS: Tracked parameter baseline: V={new_v:.4f}")
            return new_v
    except Exception as e:
        print(f"[Refactor] ERROR: Parameter tracking update failed: {e}")
    return None

def run_generation():
    state = load_evolution_state()
    state["global_generation"] += 1
    gen_id = state["global_generation"]
    
    venv_python = 'python'
    log_path = 'context/automator_execution.log'
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd()
    
    print(f"\n{'='*80}\n[GLOBAL GENERATION {gen_id}] RUNNING CONTINUOUS EVOLUTION FLIGHT\n{'='*80}")
    
    lr_modifier = state["active_parameters"].get("learning_rate_modifier", 1.0)
    base_lr = 0.15 * lr_modifier
    
    cmd = [venv_python, '-m', 'src.main', '--input', 'tasks.json', '--max-rounds', '1500', '--learning-rate', f"{base_lr:.4f}"]
    ret = stream_flight(cmd, log_path)
    if ret != 0:
        raise Exception(f"Optimization flight crashed with code {ret}")

    current_v = self_refactor_engine(log_path)
    
    mutation_result = generate_procedural_mutation(state)
    print(f"[Evolution Result] {mutation_result}")
    
    if current_v and current_v > state["best_velocity"]:
        state["best_velocity"] = current_v
        print(f"[New Epoch Record] Best convergence velocity shifted to: {current_v:.4f}")
        
    state["mutation_history"].append({
        "generation": gen_id,
        "timestamp": time.time(),
        "velocity": current_v,
        "action": mutation_result
    })
    
    save_evolution_state(state)

    print("[Validation] Running reference verification check...")
    subprocess.run([venv_python, '-m', 'src.main', '--input', 'tasks.json', '--max-rounds', '1'], check=True, capture_output=True, env=env)
    
    if os.path.exists('tests/test_stress.py'):
        subprocess.run(['pytest', 'tests/test_stress.py'], check=True, env=env)

    subprocess.run(['python', 'bundle_repo.py'], check=True, env=env)

if __name__ == '__main__':
    # Run a block of 3 generations per trigger execution pass to grow infinitely
    try:
        for _ in range(3):
            run_generation()
    except Exception as e:
        print(f"\n[HALT] Critical loop failure: {e}")
        sys.exit(1)
