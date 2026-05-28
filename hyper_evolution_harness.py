import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import subprocess
import time
import re

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

def execute_intra_run_ast_mutation(gen_id):
    """
    Generalized Macro Mutator: Dynamically targets and transforms engine 
    architectures using robust patterns that resist string variations.
    """
    print(f'\\n[AST Synthesis] Generation {gen_id} cleared. Scanning for structural code expansions...')
    engine_path = 'src/automator/substrate_engine.py'
    functional_path = 'src/automator/action_functional.py'
    
    if not os.path.exists(engine_path) or not os.path.exists(functional_path):
        return

    # Generation 1 Upgrade: Inject an adaptive clipping layer into the active optimization flow
    if gen_id == 1:
        with open(functional_path, 'r') as f: content = f.read()
        if "jnp.clip" not in content:
            print("[AST Mutation] Level 1 -> Appending functional XLA parameter clipping bounds.")
            content = re.sub(
                r"return\s+jnp\.maximum\(([^,]+),\s*floor\)",
                r"return jnp.clip(\1 + delta, floor, 12.0)",
                content
            )
            with open(functional_path, 'w') as f: f.write(content)

    # Generation 2 Upgrade: Append an autonomous Bernoulli Dropout layer into the weight mapping step
    elif gen_id == 2:
        with open(engine_path, 'r') as f: content = f.read()
        if "bernoulli" not in content:
            print("[AST Mutation] Level 2 -> Injecting functional JAX Bernoulli Dropout mask layers.")
            content = re.sub(
                r"self\.velocity_ema\s*=\s*float\(([^)]+)\)",
                r"mask = jax.random.bernoulli(jax.random.PRNGKey(int(time.time())), 0.98)\\n        self.velocity_ema = float(\1 * mask)",
                content
            )
            if "import jax" not in content:
                content = "import jax\\n" + content
            with open(engine_path, 'w') as f: f.write(content)

    # Generation 3 Upgrade: Append a structural execution dampening profile to stabilize long iterations
    elif gen_id == 3:
        with open(functional_path, 'r') as f: content = f.read()
        if "0.95" not in content and "grad" in content:
            print("[AST Mutation] Level 3 -> Appending structural decay coefficients.")
            content = re.sub(
                r"return\s+lr\s*\*+([^\\n]+)",
                r"return lr * \1 * 0.95",
                content
            )
            with open(functional_path, 'w') as f: f.write(content)

def self_refactor_engine(log_path):
    print(f'\\n[Refactor] Parsing structural trajectory logs: {log_path}')
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
    
    print(f"\\n{'='*80}\\n[MASTER GENERATION {gen_id}/5] LAUNCHING DEEP 1500-ROUND OPTIMIZATION FLIGHT\\n{'='*80}")
    
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
        print(f"\\n[CONTINUOUS TRAINING COMPLETE] Framework evolved 5 generations in {time.time()-start_time:.2f}s")
    except KeyboardInterrupt:
        print(\"\\n[HALT] Manual override detected.\")
        sys.exit(1)
    except Exception as e:
        print(f"\\n[HALT] Critical failure: {e}")
        sys.exit(1)
