# configs/base_config.py
import torch
import os

# --- 설정 변수 ---
# --- 기본 설정 ---
# 데이터 경로 (필요 시 실제 경로로 수정)
# ROOT_DIRECTORY = "path/to/your/dataset"
ROOT_DIRECTORY = "/kaggle/input/train-pill"
# 체크포인트 저장 경로
CHECKPOINT_DIR = "checkpoints"
TEST_DIRECTORY = os.path.join(ROOT_DIRECTORY, 'test_images')

# --- 훈련 설정 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

NUM_EPOCHS = 20 # or 10
BATCH_SIZE = 4 # or 2 or 8
BATCH_SIZE_VAL = 1
LEARNING_RATE = 0.005
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
MAP_CALC_CYCLE = 5

# --- 콜백 설정 ---
# EarlyStopping
ES_PATIENCE = 5
# CheckpointSaver
CS_TOP_K = 3

# --- 데이터셋 설정 ---
TRAIN_VALID_SPLIT = 0.8
NUM_WORKERS = 2 # os.cpu_count()

# 점수 임계값
SCORE_THRESHOLD = 0.5