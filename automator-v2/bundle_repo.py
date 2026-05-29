
import os

def bundle_repo(output_file='repo_context_bundle.txt'):
    repo_root = os.path.dirname(os.path.abspath(__file__))
    exclude_dirs = {'.config', 'sample_data', 'venv', '__pycache__', '.git', 'autonomous_building_framework.egg-info'}

    with open(output_file, 'w') as out:
        out.write('--- REPOSITORY CONTEXT BUNDLE ---\n')
        for root, dirs, files in os.walk(repo_root):
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            for file in files:
                # UPDATED: Added .log to allowed extensions
                if file.endswith(('.py', '.toml', '.json', '.xml', '.log')):
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, repo_root)
                    out.write(f'\n--- FILE: {rel_path} ---\n')
                    try:
                        with open(file_path, 'r') as f:
                            out.write(f.read())
                    except Exception as e:
                        out.write(f'Error reading file: {e}')
                    out.write('\n')

if __name__ == '__main__':
    bundle_repo()
    print('Repository state consolidated into repo_context_bundle.txt')
