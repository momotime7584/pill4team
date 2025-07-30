# B: 데이터 증강 강도 탐색

from utils.experiment_utils import run_grid_search

# 회전 각도와 밝기 조절 강도 탐색
grid_aug = {
    'data.augmentation.rotation_range': [15, 30],
    'data.augmentation.brightness_limit': [0.2, 0.4]
}
run_grid_search('configs/faster_rcnn_r50_fpn_baseline.py', grid_aug, "Augmentation_Search")
