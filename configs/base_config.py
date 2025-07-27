# configs/base_config.py
import torch
import os

# --- 설정 변수 ---
# --- 기본 설정 ---
# 데이터 경로 (필요 시 실제 경로로 수정)
# ROOT_DIRECTORY = "path/to/your/dataset"
# ROOT_DIRECTORY = "."
ROOT_DIRECTORY = "/kaggle/input/train-pill"

TRAIN_IMAGE_DIRECTORY = "train_images"
# 체크포인트 저장 경로
CHECKPOINT_DIR = "checkpoints"
TEST_DIRECTORY = os.path.join(ROOT_DIRECTORY, 'test_images')

# --- 훈련 설정 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# --- 재현성 설정 ---                                        
SEED = 42

NUM_EPOCHS = 30 # or 10 or 20
BATCH_SIZE = 8 # or 2 or 8 or 4
BATCH_SIZE_VAL = 1
LEARNING_RATE = 0.005 # 0.001
# LEARNING_RATE = 0.0005
ADAM_LEARNING_RATE = 0.001

MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
MAP_CALC_CYCLE = 5

# [추가] 그래디언트 클리핑에 사용할 최대 norm 값 (0 이하면 비활성화)
GRADIENT_CLIP_NORM = 5.0  # 1.0


# --- 스케줄러 설정 (ReduceLROnPlateau) ---
SCHEDULER_MODE = 'min' # 'min' 또는 'max'
SCHEDULER_FACTOR = 0.1
# --- 콜백 설정 ---
# EarlyStopping
ES_PATIENCE = 5
# CheckpointSaver

# ReduceLROnPlateau
LR_PATIENCE = 3
CS_TOP_K = 3

# --- 데이터셋 설정 ---
TRAIN_VALID_SPLIT = 0.8
NUM_WORKERS = 2 # os.cpu_count()

# [추가] 데이터셋에서 무시할 최소 박스 크기 (픽셀 단위)
MIN_BOX_SIZE = 10 

fast_rcnn_model = dict(
    type='FasterRCNN',
    backbone='ResNet50',
    pretrained=True,
    # Faster R-CNN 특화 설정
    anchor_sizes=((32,), (64,), (128,), (256,), (512,)),
    aspect_ratios=((0.5, 1.0, 2.0),),
    # RPN 설정
    rpn_pre_nms_top_n_train=2000,
    rpn_post_nms_top_n_test=1000,
    # ... (기타 rpn, roi 설정들)
    box_nms_thresh=0.5,
    box_detections_per_img=100
)

SCHEDULER_STEP_SIZE = 20 # or 30 or 5 or 20? 40?
SCHEDULER_GAMMA = 0.1

# 점수 임계값
SCORE_THRESHOLD = 0.1 # 0.5
NMS_THRESHOLD = 0.3 # 0.5
