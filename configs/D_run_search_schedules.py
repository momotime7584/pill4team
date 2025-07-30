# D: 훈련 길이 및 스케줄러 탐색

from utils.experiment_utils import run_grid_search

grid_schedule = {
    'training.num_epochs': [12, 20],
    'lr_scheduler.type': ['StepLR', 'CosineAnnealingLR']
}
run_grid_search('configs/faster_rcnn_r50_fpn_baseline.py', grid_schedule, "Schedule_Search")