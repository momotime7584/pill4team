# C: 모델 파라미터 탐색

from utils.experiment_utils import run_grid_search

# 모델 내부의 점수 임계값 탐색
grid_model = {
    'model.roi_head.score_thresh': [0.05, 0.1],
    'training.num_epochs': [3] # 짧은 에폭
}
run_grid_search('configs/faster_rcnn_r50_fpn_baseline.py', grid_model, "Model_Param_Search")
