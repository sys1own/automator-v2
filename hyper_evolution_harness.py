import subprocess, os, re, time
from src.automator.expansion_manager import ExpansionManager

def run_evolutionary_flight():
    manager = ExpansionManager()
    log_path = '/content/context/automator_execution.log'
    
    for gen in range(1, 6):
        print(f'\n=== STARTING GENERATION {gen}/5 ===')
        
        # Execute flight
        cmd = ['/content/venv/bin/python', 'src/main.py', '--input', 'tasks.json', '--max-rounds', '100']
        subprocess.run(cmd, check=True)
        
        # Structural Synthesis: Evolve the engine logic between generations
        print(f'[Synthesis] Mutating AST for Generation {gen+1} optimization...')
        manager.manufacture_module('/content/src/automator/substrate_engine.py', 'substrate_engine')
        
        # Archiving
        subprocess.run(['python', 'bundle_repo.py'], check=True)

if __name__ == "__main__":
    run_evolutionary_flight()