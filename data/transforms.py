# data/transforms.py
import torch
import torchvision.transforms.v2 as T

# data/transforms.py
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2 # albumentations는 내부적으로 OpenCV를 사용합니다

# --- 이 부분을 config 파일로 옮겨서 관리하면 더 좋습니다 ---
IMG_SIZE = 512 # 모델에 입력할 이미지 크기
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
# ---

# Albumentations는 bbox 형식을 지정해야 합니다.
# 우리는 [x_min, y_min, x_max, y_max] 형식을 사용하므로 'pascal_voc' 입니다.
BBOX_PARAMS = A.BboxParams(format='pascal_voc', label_fields=['labels'])


def get_transform(train: bool):
    """
    Albumentations를 사용한 데이터 변환 파이프라인을 반환합니다.

    Args:
        train (bool): 훈련 모드인 경우 True, 아닐 경우 False.
    """
    if train:
        # 훈련용 데이터 증강 파이프라인
        return A.Compose([
            # --- 기하학적 변환 ---
            # A.HorizontalFlip(p=0.5),
            # # A.VerticalFlip(p=0.5), # 알약의 위아래가 중요할 수 있으므로 선택적으로 사용
            # A.Rotate(limit=30, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0),
            
            # # --- 크기 및 위치 변환 ---
            # # 이미지를 모델 입력 크기에 맞게 리사이즈합니다.
            # # 작은 객체가 잘려나가지 않도록 LongestMaxSize를 사용하는 것이 안전할 수 있습니다.
            # A.LongestMaxSize(max_size=IMG_SIZE, p=1.0),
            # A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE, p=1.0, border_mode=cv2.BORDER_CONSTANT, value=0),

            # # --- 색상 변환 ---
            # A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.75),
            # A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.5),
            # A.ISONoise(p=0.3),
            # A.GaussNoise(p=0.3),

            # --- 최종 변환 ---
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ], bbox_params=BBOX_PARAMS)
    else:
        # 검증/테스트용 변환 파이프라인 (증강 없음)
        return A.Compose([
            # A.LongestMaxSize(max_size=IMG_SIZE, p=1.0),
            # A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE, p=1.0, border_mode=cv2.BORDER_CONSTANT, value=0),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ], bbox_params=BBOX_PARAMS)

def get_basic_transform():
    transforms = []
    transforms.append(T.ToImage())
    transforms.append(T.ToDtype(torch.float32, scale=True))
    return T.Compose(transforms)


def get_transform_old(train: bool):
    transforms = []
    transforms.append(T.ToImage())
    transforms.append(T.ToDtype(torch.float32, scale=True))
    # if train:
    #     transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def get_augmented_transform():
    # 추후 이 곳에 회전, 밝기 조절, CopyPaste 등의 증강 로직을 추가합니다.
    pass

