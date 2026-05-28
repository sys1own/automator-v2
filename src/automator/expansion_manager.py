import os
import json
import ast

class ExpansionManager:
    def __init__(self, src_root='/content/src', task_json='/content/tasks.json'):
        self.src_root = src_root
        self.task_json = task_json
        # Strictly enforce reserved names to protect environment packages from shadowing loops
        self.reserved_names = {
            'ast', 'argparse', 'random', 'hashlib', 'math', 'threading',
            'subprocess', 'signal', 'psutil', 'curses', 'multiprocessing',
            'os', 'sys', 'time', 'struct', 'json', 'logging', 'inspect', 'typing',
            'numpy', 'scipy', 'pandas', 'pytest', 'jax', 'jnp', 'jaxlib', 'importlib'
        }

    def check_architectural_gaps(self, dependency_graph):
        for edge in dependency_graph:
            target = edge['to']
            parts = target.split('.')
            while parts and parts[0] == 'src':
                parts = parts[1:]

            if not parts or parts[0] in self.reserved_names: 
                continue

            target_path = os.path.join(self.src_root, *parts) + '.py'

            if not os.path.exists(target_path):
                print(f'[Expansion] Dynamic gap detected: {target}. Manufacturing...')
                self.manufacture_module(target_path, target)

    def manufacture_module(self, path, module_name):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        class_name = "".join(x.capitalize() for x in module_name.split('.')[-1].split('_'))

        code_template = f'''# Autonomous Module: {module_name}
import numpy as np

class {class_name}:
    def __init__(self):
        self.identity = "{module_name}"
        print(f"[{class_name}] Substrate Hook Activated.")

    def execute_logic(self, frame_data):
        return np.tanh(frame_data) * 0.99
'''
        with open(path, 'w') as f:
            f.write(code_template)
        self.register_in_task_matrix(path)

    def register_in_task_matrix(self, full_path):
        rel_path = os.path.relpath(full_path, '/content')
        with open(self.task_json, 'r') as f:
            data = json.load(f)

        if not any(t['scope'] == rel_path for t in data['tasks']):
            data['tasks'].append({
                "scope": rel_path,
                "candidates": [
                    {"id": "optimized_path", "features": ["jit", "vectorized"], "base_weight": 0.9},
                    {"id": "stable_path", "features": ["eager"], "base_weight": 1.0}
                ]
            })
            with open(self.task_json, 'w') as f:
                json.dump(data, f, indent=4)
            print(f'[Expansion] Registered {rel_path} in task matrix.')
