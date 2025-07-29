# configs/_base_/default_runtime.py
import torch
import random
import numpy as np

# 전체 훈련 워크플로우
training = dict(
    num_epochs=20, # '1x' 스케줄은 보통 12 에폭
    map_calc_cycle=1, # 매 에폭마다 mAP 계산
    grad_clip_norm=1.0
)

# 검증 주기 (훈련 흐름 제어)
validation = dict(map_calc_cycle=1)

# 콜백 (훈련 흐름 제어)
# EarlyStopping은 총 에폭 수에 따라 patience가 달라질 수 있으므로,
# 스케줄 파일에 있는 것이 더 적합할 수도 있습니다. (설계 선택)

callbacks = dict(
    early_stopping=dict(
        patience=5
    ),
    checkpoint_saver=dict(
        top_k=3, 
        save_dir='checkpoints'
    )
)

# 1. 재현성 및 환경 설정
seed = 42
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 2. 데이터 로더 설정 (훈련 방법과 무관한 실행 환경)
data_loaders = dict(
    num_workers=2
)

# 3. 로깅 및 체크포인트 (실험 관리 도구)
checkpoint_saver = dict(
    top_k=3, 
    save_dir='checkpoints'
)

# --- MLOps 설정 ---
mlflow = dict(
    tracking_uri="mlruns", # MLflow 데이터를 저장할 로컬 디렉토리
    experiment_name="Pill_Detection_Experiment"
)
