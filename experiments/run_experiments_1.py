# tools/run_experiments.py
from utils.experiment_utils import run_grid_search

def main():
    # --- 실험 그룹 1: SGD의 학습률과 배치 사이즈 탐색 ---
    grid1 = {
        'optimizer.lr': [0.01, 0.005],
        'data.train_batch_size': [8, 16],
    }
    run_grid_search(
        base_config_path='configs/faster_rcnn_r50_fpn_1x_pill_sgd.py',
        param_grid=grid1,
        experiment_group_name="SGD_LR_BS_Search"
    )
    
    # --- 실험 그룹 2: AdamW 옵티마이저 실험 ---
    grid2 = {
        'optimizer.type': ['AdamW'],
        'optimizer.lr': [0.0001, 0.00005],
    }
    run_grid_search(
        base_config_path='configs/faster_rcnn_r50_fpn_1x_pill_sgd.py',
        param_grid=grid2,
        experiment_group_name="AdamW_Search"
    )

if __name__ == '__main__':
    main()