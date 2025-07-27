# utils/seed_utils.py
import torch
import numpy as np
import random
import os

def set_seed(seed_value):
    """
    모든 무작위 연산의 시드를 고정하여 재현성을 확보합니다.
    """
    random.seed(seed_value) # Python random 모듈
    np.random.seed(seed_value) # NumPy
    torch.manual_seed(seed_value) # PyTorch CPU

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value) # PyTorch GPU (단일 GPU)
        torch.cuda.manual_seed_all(seed_value) # PyTorch GPU (멀티 GPU)

    # cuDNN 결정론적 모드 설정 (성능 저하 가능성 있음)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # deterministic이 True일 때 False 권장

    # 환경 변수 설정 (일부 라이브러리에서 사용)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

def worker_init_fn(worker_id):
    """
    DataLoader 워커 프로세스 내에서 시드를 고정하기 위한 함수.
    """
    # 각 워커마다 고유한 시드를 부여하여 충돌 방지
    # DataLoader의 seed는 메인 프로세스의 seed + worker_id로 설정됨
    # 따라서 여기에 다시 seed를 설정할 필요는 없지만,
    # 만약 워커 내에서 추가적인 random/numpy/torch 연산이 있다면 필요할 수 있음.
    # 여기서는 DataLoader의 내부 시드 메커니즘을 따르도록 합니다.
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    # torch.manual_seed(worker_seed) # 이미 torch.initial_seed()로 설정됨
    # torch.cuda.manual_seed_all(worker_seed) # 이미 torch.initial_seed()로 설정됨
