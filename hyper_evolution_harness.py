import os
import subprocess
import sys
import time
import re

def stream_flight(cmd, log_path):
    """High-throughput line-streaming multiplexer to log telemetry bounds."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, 'w') as log_file: # Fresh file per pass to conserve tokens
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

def self_refactor_engine(log_path):
    """Parses metrics and updates substrate_engine.py class defaults natively."""
    print(f'\n[Refactor] Parsing structural trajectory logs: {log_path}')
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
            print(f"[Refactor] SUCCESS: Hardcoded engine parameters: V={new_v}, D={new_d}")
    except Exception as e:
        print(f"[Refactor] ERROR: Mechanical parameters update failed: {e}")

def run_generation(gen_id):
    venv_python = 'python'
    log_path = 'context/automator_execution.log'
    
    print(f"\n{'='*80}\n[MASTER GENERATION {gen_id}/5] LAUNCHING DEEP 1500-ROUND OPTIMIZATION FLIGHT\n{'='*80}")
    
    # 1. High-Throughput Matrix Execution Sweep (Runs for 2-3 minutes per generation)
    cmd = [venv_python, 'src/main.py', '--input', 'tasks.json', '--max-rounds', '1500', '--learning-rate', '0.15']
    ret = stream_flight(cmd, log_path)
    if ret != 0:
        raise Exception(f"Optimization flight crashed with code {ret}")

    # 2. Automated Self-Refactoring
    self_refactor_engine(log_path)

    # 3. Systems Validation Passes
    print("[Validation] Running ExpansionManager and reference boundary diagnostics...")
    subprocess.run([venv_python, 'src/main.py', '--input', 'tasks.json', '--max-rounds', '1'], check=True, capture_output=True)
    
    if os.path.exists('tests/test_stress.py'):
        print("[Validation] Running stress regression tests...")
        subprocess.run(['pytest', 'tests/test_stress.py'], check=True)

    # 4. Freeze Token-Compressed State
    print("[Validation] Compressing and archiving updated codebase layout...")
    subprocess.run(['python', 'bundle_repo.py'], check=True)

if __name__ == '__main__':
    start_time = time.time()
    try:
        for g in range(1, 6):
            run_generation(g)
        print(f"\n[CONTINUOUS TRAINING COMPLETE] Framework evolved 5 generations in {time.time()-start_time:.2f}s")
    except KeyboardInterrupt:
        print("\n[HALT] Manual override detected.")
    except Exception as e:
        print(f"\n[HALT] Critical failure: {e}")
