import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import subprocess
import time
import re
import ast
import shutil
import tempfile

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

def _gate1_static_verification(code, label):
    """Gate 1 -- formal static verification via ast.parse + compile.

    Rejects SyntaxError / IndentationError (IndentationError is a subclass of
    SyntaxError) before anything executes. A NameError is a runtime fault that
    the compiler cannot detect, so it is deliberately deferred to Gate 2.
    """
    try:
        ast.parse(code)
        compile(code, label, "exec")
        return True, None
    except (SyntaxError, ValueError) as exc:
        return False, exc


def _gate2_dynamic_sandbox(target_rel, code, root, rounds=2):
    """Gate 2 -- dynamic sandbox execution in throwaway isolation.

    Copies the project into a temp directory, swaps in the candidate module,
    and runs the real controller for `rounds` rounds. The production tree is
    never touched. Passing requires exit code 0, at least `rounds` async VDF
    verification frames, and no FATAL verification failures.
    """
    sandbox = tempfile.mkdtemp(prefix="guardian_")
    try:
        shutil.copytree(os.path.join(root, "src"), os.path.join(sandbox, "src"),
                        ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copy(os.path.join(root, "tasks.json"), os.path.join(sandbox, "tasks.json"))
        with open(os.path.join(sandbox, target_rel), "w") as fh:
            fh.write(code)
        env = os.environ.copy()
        env["PYTHONPATH"] = sandbox
        proc = subprocess.run(
            ["python", "-m", "src.main", "--input", "tasks.json", "--max-rounds", str(rounds)],
            cwd=sandbox, env=env, capture_output=True, text=True, timeout=180
        )
        frames = proc.stdout.count("verified asynchronously")
        passed = proc.returncode == 0 and frames >= rounds and "FATAL" not in proc.stdout
        return passed, proc, frames
    except subprocess.TimeoutExpired as exc:
        return False, exc, 0
    finally:
        shutil.rmtree(sandbox, ignore_errors=True)


def guarded_graduate(prod_path, mutated_code, rounds=2):
    """Double-Gate Guardian Isolation Protocol for source self-mutation.

    Writes the candidate to a quarantine file ('<module>.tmp'), then requires
    BOTH Gate 1 (static ast/compile) and Gate 2 (isolated dynamic flight) to
    approve before graduating it onto the production module via an atomic
    os.replace(). If either gate rejects, the production file is left
    byte-for-byte intact and the quarantine file is removed. Returns True only
    when the mutation is graduated.
    """
    root = os.path.dirname(os.path.abspath(__file__))
    prod_abs = os.path.abspath(prod_path)
    target_rel = os.path.relpath(prod_abs, root)
    quarantine = prod_abs + ".tmp"

    with open(quarantine, "w") as fh:
        fh.write(mutated_code)
    try:
        ok1, err1 = _gate1_static_verification(mutated_code, quarantine)
        if not ok1:
            print("[Guardian] GATE 1 (static) REJECTED %s: %s: %s"
                  % (target_rel, type(err1).__name__, err1))
            return False
        print("[Guardian] Gate 1 (ast.parse/compile) PASSED for %s" % target_rel)

        ok2, proc, frames = _gate2_dynamic_sandbox(target_rel, mutated_code, root, rounds)
        if not ok2:
            exit_code = getattr(proc, "returncode", "TIMEOUT")
            print("[Guardian] GATE 2 (sandbox flight) REJECTED %s: exit=%s vdf_frames=%s"
                  % (target_rel, exit_code, frames))
            stderr_tail = (getattr(proc, "stderr", "") or "")[-600:]
            if stderr_tail:
                print(stderr_tail)
            return False
        print("[Guardian] Gate 2 (isolated %d-round flight) PASSED for %s: exit=0, %d VDF frames"
              % (rounds, target_rel, frames))

        os.replace(quarantine, prod_abs)
        print("[Guardian] GRADUATED mutation into %s via atomic move" % target_rel)
        return True
    finally:
        if os.path.exists(quarantine):
            os.remove(quarantine)
            print("[Guardian] Quarantine cleaned: %s.tmp" % target_rel)


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
            guarded_graduate(functional_path, content)

    # Generation 2 Upgrade: Append an autonomous Bernoulli Dropout layer into the weight mapping step
    elif gen_id == 2:
        with open(engine_path, 'r') as f: content = f.read()
        if "bernoulli" not in content:
            print("[AST Mutation] Level 2 -> Injecting functional JAX Bernoulli Dropout mask layers.")
            content = re.sub(
                r"self\.velocity_ema\s*=\s*float\(([^)]+)\)",
                r"mask = jax.random.bernoulli(jax.random.PRNGKey(int(np.random.randint(0, 100000))), 0.98)\n        self.velocity_ema = float(\1 * mask)",
                content
            )
            if "import jax\n" not in content:
                content = "import jax\n" + content
            guarded_graduate(engine_path, content)

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
            guarded_graduate(functional_path, content)

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
            
            if guarded_graduate(engine_path, content):
                print(f"[Refactor] SUCCESS: Hardcoded engine parameters: V={new_v}")
            else:
                print(f"[Refactor] ABORTED: guardian rejected V={new_v}; production preserved")
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
        print("\n[HALT] Manual override detected.")
        sys.exit(1)
    except Exception as e:
        print(f"\\n[HALT] Critical failure: {e}")
        sys.exit(1)
