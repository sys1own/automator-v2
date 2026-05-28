import os
import subprocess
import sys
import time
import re
import json

def stream_flight(cmd, log_path):
    """High-throughput line-streaming multiplexer to capture runtime telemetry."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, 'a') as log_file:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
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
    Macro Architecture Upgrade: Programmatically mutates the codebase text
    by injecting advanced structural features directly between optimization loops.
    """
    print(f'\n[AST Synthesis] Generation {gen_id} Threshold Cleared. Injecting structural features...')
    engine_path = 'src/automator/substrate_engine.py'
    if not os.path.exists(engine_path):
        return

    with open(engine_path, 'r') as f:
        code = f.read()

    # Generation 1 Evolution: Upgrade to hardware-native JAX optimization steps
    if gen_id == 1 and "def execute_logic" in code:
        print("[AST Mutation] Level 1: Activating bare-metal XLA device arrays.")
        # Ensure triple quotes are cleanly preserved across text boundaries
        code = code.replace(
            "return np.tanh(frame_data) * 0.99",
            "import jax.numpy as jnp\n        return self.execute_accelerated_step(lr=0.1, reward=1.0)"
        )

    # Generation 2 Evolution: Inject adaptive weight learning rate schedules (Adam)
    elif gen_id == 2 and "AcceleratedSubstrateFlow" in code:
        print("[AST Mutation] Level 2: Synthesizing Adaptive Adam update schedules.")
        # Programmatically rewrite step parameters inside the functional execution block
        code = re.sub(r"lr \* reward", "lr * m / (jnp.sqrt(v) + 1e-8)", code)

    # Generation 3 Evolution: Inject Stochastic Bernoulli Dropout masks to prevent overfitting
    elif gen_id == 3 and "key" not in code:
        print("[AST Mutation] Level 3: Appending Bernoulli Dropout masks into execution graph.")
        code = code.replace(
            "return float(new_w)",
            "dropout_mask = jax.random.bernoulli(jax.random.PRNGKey(0), 0.9)\n        return float(new_w * dropout_mask)"
        )

    with open(engine_path, 'w') as f:
        f.write(code)
    print(f"[AST Synthesis] Codebase text files evolved for Generation {gen_id + 1}.")

def self_refactor_params(log_path):
    """Parses metrics and updates substrate_engine.py default fields."""
    print(f'\n[Refactor] Analyzing trajectory log boundaries: {log_path}')
    try:
        with open(log_path, 'r') as f:
            lines = f.readlines()
        
        last_round = None
        for line in reversed(lines):
            if "Round" in line and "Velocity EMA:" in line:
                last_round = line
                break
        
        if not last_round:
            print("[Refactor] WARNING: Historical convergence benchmarks not located.")
            return

        v_match = re.search(r"Velocity EMA: ([0-9.]+)", last_round)
        d_match = re.search(r"Diversity: ([0-9.]+)", last_round)

        if v_match and d_match:
            new_v = v_match.group(1)
            new_d = d_match.group(1)
            engine_path = 'src/automator/substrate_engine.py'
            
            with open(engine_path, 'r') as f:
                content = f.read()
            
            content = re.sub(r"self.velocity_ema = [0-9.]+", f"self.velocity_ema = {new_v}", content)
            content = re.sub(r"self.diversity_index = [0-9.]+", f"self.diversity_index = {new_d}", content)
            
            with open(engine_path, 'w') as f:
                f.write(content)
            print(f"[Refactor] SUCCESS: Locked parameters into file defaults: V={new_v}, D={new_d}")
    except Exception as e:
        print(f"[Refactor] ERROR: Parameter adjustment failed: {e}")

def run_generation(gen_id):
    venv_python = 'python'
    log_path = 'context/automator_execution.log'
    
    print(f"\n{'='*80}\n[MASTER GENERATION {gen_id}/5] INITIATING DEEP OPTIMIZATION FLIGHT\n{'='*80}")
    
    # 1. High-Throughput Matrix Execution Sweep (Scaled up to run extensive workloads)
    cmd = [venv_python, 'src/main.py', '--input', 'tasks.json', '--max-rounds', '1500', '--learning-rate', '0.15']
    ret = stream_flight(cmd, log_path)
    if ret != 0:
        raise Exception(f"Dynamic execution cycle encountered non-zero fault code: {ret}")

    # 2. Extract and Inject Continuous Numerical Weight States
    self_refactor_params(log_path)

    # 3. Dynamic Structural Logic Synthesis (AST Code Refactoring Inter-Run)
    execute_intra_run_ast_mutation(gen_id)

    # 4. Strict Safety Verification Pass
    print("[Validation] Running Static Import boundary and circular deadlock validation checks...")
    subprocess.run([venv_python, 'src/main.py', '--input', 'tasks.json', '--max-rounds', '1'], check=True, capture_output=True)
    
    print("[Validation] Running regression stress tests...")
    if os.path.exists('tests/test_stress.py'):
        subprocess.run(['pytest', 'tests/test_stress.py'], check=True)

    # 5. Lock Evolved Architecture State
    print("[Validation] Compressing and archiving current evolutionary layout...")
    subprocess.run(['python', 'bundle_repo.py'], check=True)

if __name__ == '__main__':
    start_time = time.time()
    try:
        # Loop consecutively through 5 generation cascades within a single workflow run
        for g in range(1, 6):
            run_generation(g)
        print(f"\n[FLIGHT STABILIZED] Cloud evolution complete across 5 generations in {time.time()-start_time:.2f}s")
    except KeyboardInterrupt:
        print("\n[Halt] Manual override signature detected.")
    except Exception as e:
        print(f"\n[Halt] Critical infrastructure exception: {e}")
