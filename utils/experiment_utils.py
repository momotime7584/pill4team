# utils/experiment_utils.py
import subprocess
import itertools

def run_grid_search(base_config_path, param_grid, experiment_group_name):
    keys, values = zip(*param_grid.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print(f"--- {experiment_group_name} 그룹: 총 {len(experiments)}개 실험 시작 ---")

    for i, params in enumerate(experiments):
        run_name = f"{experiment_group_name}_{i+1}"
        
        command = [
            'python', 'tools/train.py',
            base_config_path,
            '--run_name', run_name
        ]
        
        options = [f"{key}={value}" for key, value in params.items()]
        if options:
            command.append('--options')
            command.extend(options)
        
        print(f"\n[실행 {i+1}/{len(experiments)}] {' '.join(command)}")
        subprocess.run(command)