# A: 학습률 & 옵티마이저 탐색

from utils.experiment_utils import run_grid_search

# SGD의 LR 탐색
grid_sgd = {
    'optimizer.lr': [0.005, 0.002, 0.001],
    'training.num_epochs': [3] # 짧은 에폭
}
run_grid_search('configs/faster_rcnn_r50_fpn_baseline.py', grid_sgd, "SGD_LR_Search")

# AdamW 탐색
grid_adamw = {
    'optimizer.type': ['AdamW'], 
    'optimizer.lr': [0.0001, 0.00005],
    'training.num_epochs': [3] # 짧은 에폭
}
run_grid_search('configs/faster_rcnn_r50_fpn_baseline.py', grid_adamw, "AdamW_Search")
