# A: 학습률 & 옵티마이저 탐색

from utils.experiment_utils import run_grid_search

# SGD의 LR 탐색
grid_sgd = {
    'optimizer.type': ['SGD', 'AdamW'],
    'optimizer.lr': [0.01, 0.001, 0.0001], # 넓은 LR 범위
    'training.num_epochs': [3] # 짧은 에폭
}
run_grid_search('configs/faster_rcnn_r50_fpn_baseline.py', grid_sgd, "Phase1_Optimizer_Search_3")
