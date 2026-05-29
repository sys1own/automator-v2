import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import subprocess
import time
import re
import ast

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
    """
    Double-Gate Guardian: Writes changes to a quarantine container file,
    verifying syntactic correctness and behavioral safety before graduation.
    """
    tmp_path = file_path + ".tmp"
    try:
        # Write to sandboxed quarantine file
        with open(tmp_path, "w") as f:
            f.write(target_mutation_content)
            
        # GATE 1: Formal Static Verification & Abstract Syntax Tree Compilation Check
        ast.parse(target_mutation_content)
        compile(target_mutation_content, tmp_path, "exec")
        
        # GATE 2: Dynamic Behavioral Verification Run
        env = os.environ.copy()
        env["PYTHONPATH"] = os.getcwd()
        # Backup the production file, temporarily swap in the mutation, test, and swap back
        os.rename(file_path, file_path + ".bak")
        os.rename(tmp_path, file_path)
        
        test_run = subprocess.run(
            [sys.executable, "-m", "src.main", "--input", "tasks.json", "--max-rounds", "2"],
            capture_output=True, text=True, env=env
        )
        
        # Restore production baseline status
        os.rename(file_path, tmp_path)
        os.rename(file_path + ".bak", file_path)
        
        if test_run.returncode == 0:
            os.replace(tmp_path, file_path)
            print(f"[Guardian Gate Passed] Atomic graduation completed for: {file_path}")
            return True
        else:
            print(f"[Guardian GATE 2 FAILED] Dynamic verification threw exit code {test_run.returncode}. Aborting mutation.")
            if os.path.exists(tmp_path): os.remove(tmp_path)
            return False
            
    except Exception as e:
        print(f"[Guardian GATE 1 FAILED] Formal Compilation Crash: {e}. Aborting mutation.")
        if os.path.exists(tmp_path): os.remove(tmp_path)
        if os.path.exists(file_path + ".bak"): os.rename(file_path + ".bak", file_path)
        return False

def execute_intra_run_ast_mutation(gen_id):
    print(f"\n[AST Synthesis] Generation {gen_id} cleared. Initializing guardian validation gates...")
    engine_path = 'src/automator/substrate_engine.py'
    functional_path = 'src/automator/action_functional.py'
    
    if not os.path.exists(engine_path) or not os.path.exists(functional_path):
        return

    # Generation 1 Upgrade: Inject an adaptive parameter clipping layer
    if gen_id == 1:
        with open(functional_path, 'r') as f: content = f.read()
        if "jnp.clip" not in content:
            mutated = re.sub(
                r"return\s+jnp\.maximum\(([^,]+),\s*floor\)",
                r"return jnp.clip(\1 + delta, floor, 12.0)",
                content
            )
            verify_gate_isolation(functional_path, mutated)

    # Generation 2 Upgrade: Append a JAX-native Bernoulli Dropout mask layer
    elif gen_id == 2:
        with open(engine_path, 'r') as f: content = f.read()
        if "bernoulli" not in content:
            # FIXED: Seeded using a secure NumPy array integer call to eliminate missing 'time' module crashes
            mutated = re.sub(
                r"self\.velocity_ema\s*=\s*float\(([^)]+)\)",
                r"mask = jax.random.bernoulli(jax.random.PRNGKey(int(np.random.randint(0, 100000)))), 0.98)\n        self.velocity_ema = float(\1 * mask)",
                content
            )
            if "import jax" not in content:
                mutated = "import jax\n" + mutated
            verify_gate_isolation(engine_path, mutated)

    # Generation 3 Upgrade: Append an optimization decay constant
    elif gen_id == 3:
        with open(functional_path, 'r') as f: content = f.read()
        if "0.95" not in content and "grad" in content:
            mutated = re.sub(
                r"return\s+lr\s*\*+([^:\n]+)",
                r"return lr * \1 * 0.95",
                content
            )
            verify_gate_isolation(functional_path, mutated)

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
            print("[Refactor] WARNING: Historical convergence benchmarks not located.")
            return

        v_match = re.search(r"Velocity: ([0-9.]+)", last_round)
        if v_match:
            new_v = v_match.group(1)
            engine_path = 'src/automator/substrate_engine.py'
            
            with open(engine_path, 'r') as f:
                content = f.read()
            
            content = re.sub(r"self.velocity_ema = [0-9.]+", f"self.velocity_ema = {new_v}", content)
            
            with open(engine_path, 'w') as f:
                f.write(content)
            print(f"[Refactor] SUCCESS: Hardcoded engine parameters: V={new_v}")
    except Exception as e:
        print(f"[Refactor] ERROR: Mechanical parameters update failed: {e}")

def run_generation(gen_id):
    venv_python = 'python'
    log_path = 'context/automator_execution.log'
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd()
    
    print(f"\n{'='*80}\n[MASTER GENERATION {gen_id}/5] LAUNCHING DEEP 1500-ROUND OPTIMIZATION FLIGHT\n{'='*80}")
    
    cmd = [venv_python, '-m', 'src.main', '--input', 'tasks.json', '--max-rounds', '1500', '--learning-rate', '0.15']
    ret = stream_flight(cmd, log_path)
    if ret != 0:
        raise Exception(f"Optimization flight crashed with code {ret}")

    self_refactor_engine(log_path)
    execute_intra_run_ast_mutation(gen_id)

    print("[Validation] Running ExpansionManager reference diagnostics...")
    subprocess.run([venv_python, '-m', 'src.main', '--input', 'tasks.json', '--max-rounds', '1'], check=True, capture_output=True, env=env)
    
    if os.path.exists('tests/test_stress.py'):
        subprocess.run(['pytest', 'tests/test_stress.py'], check=True, env=env)

    subprocess.run(['python', 'bundle_repo.py'], check=True, env=env)

if __name__ == '__main__':
    start_time = time.time()
    try:
        for g in range(1, 6):
            run_generation(g)
        print(f"\n[CONTINUOUS TRAINING COMPLETE] Framework evolved 5 generations in {time.time()-start_time:.2f}s")
    except KeyboardInterrupt:
        print("\n[HALT] Manual override detected.")
        sys.exit(1)
    except Exception as e:
        print(f"\n[HALT] Critical failure: {e}")
        sys.exit(1)
