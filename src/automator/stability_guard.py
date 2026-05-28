
import json
import os

class StabilityGuard:
    @staticmethod
    def rollback_and_prune(task_json='/content/tasks.json'):
        print('[StabilityGuard] CRITICAL FAILURE DETECTED. Initiating pruning...')
        if not os.path.exists(task_json): return
        
        with open(task_json, 'r') as f:
            data = json.load(f)
        
        # Penalize weights for the most recently added or modified tasks
        if data.get('tasks'):
            for candidate in data['tasks'][-1].get('candidates', []):
                if candidate['id'] == 'optimized_path':
                    candidate['base_weight'] = max(0.1, candidate['base_weight'] - 0.5)
        
        with open(task_json, 'w') as f:
            json.dump(data, f, indent=4)
        print('[StabilityGuard] Variant pruned. Negative reward re-routed to bandit matrices.')
