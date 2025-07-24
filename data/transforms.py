# data/transforms.py
import torch
import torchvision.transforms.v2 as T

# data/transforms.py
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2 # albumentations는 내부적으로 OpenCV를 사용합니다

# 아래에 정의된 transform들은 모두 아래 MEAN과 STD를 사용하여 normalization을 수행합니다.
IMG_SIZE = 512 # 모델에 입력할 이미지 크기
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
# ---

# Albumentations는 bbox 형식을 지정해야 합니다.
# 우리는 [x_min, y_min, x_max, y_max] 형식을 사용하므로 'pascal_voc' 입니다.
BBOX_PARAMS = A.BboxParams(format='pascal_voc', label_fields=['labels'])

#아래에 정의된 transform들 중 get_A_transform4가 가장 높은 validation mAP를 기록했습니다. (submission mAP: 0.9628)
#참고) augmentation을 적용하지 않은 경우 validation mAP: 0.8629로 normalization만으로도 좋은 결과를 기록합니다.

#vertical flip(p=0.5)을 수행합니다. validation mAP: 0.8208
def get_A_transform1(train: bool):
    """
    Albumentations를 사용한 데이터 변환 파이프라인을 반환합니다.

    Args:
        train (bool): 훈련 모드인 경우 True, 아닐 경우 False.
    """
    if train:
        # 훈련용 데이터 증강 파이프라인
        return A.Compose([
            # --- 기하학적 변환 ---
            A.VerticalFlip(p=0.5), # 알약의 위아래가 중요할 수 있으므로 선택적으로 사용

            # --- 최종 변환 ---
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ], bbox_params=BBOX_PARAMS)
    else:
        # 검증/테스트용 변환 파이프라인 (증강 없음)
        return A.Compose([
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ], bbox_params=BBOX_PARAMS)

#horizontal flip(p=0.5)을 수행합니다. validation mAP: 0.7814
def get_A_transform2(train: bool):
    """
    Albumentations를 사용한 데이터 변환 파이프라인을 반환합니다.

    Args:
        train (bool): 훈련 모드인 경우 True, 아닐 경우 False.
    """
    if train:
        # 훈련용 데이터 증강 파이프라인
        return A.Compose([
            # --- 기하학적 변환 ---
            A.HorizontalFlip(p=0.5),

            # --- 최종 변환 ---
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ], bbox_params=BBOX_PARAMS)
    else:
        # 검증/테스트용 변환 파이프라인 (증강 없음)
        return A.Compose([
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ], bbox_params=BBOX_PARAMS)

#vertical flip(p=0.5)와 horizontal flip(p=0.3)을 수행합니다. validation mAP: 0.4960
def get_A_transform3(train: bool):
    """
    Albumentations를 사용한 데이터 변환 파이프라인을 반환합니다.

    Args:
        train (bool): 훈련 모드인 경우 True, 아닐 경우 False.
    """
    if train:
        # 훈련용 데이터 증강 파이프라인
        return A.Compose([
            # --- 기하학적 변환 ---
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.3),

            # --- 최종 변환 ---
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ], bbox_params=BBOX_PARAMS)
    else:
        # 검증/테스트용 변환 파이프라인 (증강 없음)
        return A.Compose([
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ], bbox_params=BBOX_PARAMS)

#rotation(p=0.5)을 수행합니다. validation mAP: 0.8727
def get_A_transform4(train: bool):
    """
    Albumentations를 사용한 데이터 변환 파이프라인을 반환합니다.

    Args:
        train (bool): 훈련 모드인 경우 True, 아닐 경우 False.
    """
    if train:
        # 훈련용 데이터 증강 파이프라인
        return A.Compose([
            # --- 기하학적 변환 ---
            A.Rotate(limit=30, p=0.5, border_mode=cv2.BORDER_REPLICATE),

            # --- 최종 변환 ---
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ], bbox_params=BBOX_PARAMS)
    else:
        # 검증/테스트용 변환 파이프라인 (증강 없음)
        return A.Compose([
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ], bbox_params=BBOX_PARAMS)

#rotation(p=0.5)과 vertical flip(p=0.5)을 수행합니다. validation mAP: 0.7632
def get_A_transform5(train: bool):
    """
    Albumentations를 사용한 데이터 변환 파이프라인을 반환합니다.

    Args:
        train (bool): 훈련 모드인 경우 True, 아닐 경우 False.
    """
    if train:
        # 훈련용 데이터 증강 파이프라인
        return A.Compose([
            # --- 기하학적 변환 ---
            A.Rotate(limit=30, p=0.5, border_mode=cv2.BORDER_REPLICATE, value=0),
            A.VerticalFlip(p=0.5),

            # --- 최종 변환 ---
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ], bbox_params=BBOX_PARAMS)
    else:
        # 검증/테스트용 변환 파이프라인 (증강 없음)
        return A.Compose([
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ], bbox_params=BBOX_PARAMS)

#rotation 시 빈 공간을 파란색으로 채우도록 조건을 변경했습니다. (border_mode=cv2.BORDER_CONSTANT, value=(0, 128, 255))
#그러나 샘플이미지 출력 시 위의 4번 transform과 동일해서 학습시키지 않았습니다.
def get_A_transform6(train: bool):
    """
    Albumentations를 사용한 데이터 변환 파이프라인을 반환합니다.

    Args:
        train (bool): 훈련 모드인 경우 True, 아닐 경우 False.
    """
    if train:
        # 훈련용 데이터 증강 파이프라인
        return A.Compose([
            # --- 기하학적 변환 ---
            A.Rotate(limit=30, p=0.5, border_mode=cv2.BORDER_CONSTANT,
            value=(0, 128, 255)),  # 파란 배경 RGB (혹은 실제 배경 평균값)

            # --- 최종 변환 ---
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ], bbox_params=BBOX_PARAMS)
    else:
        # 검증/테스트용 변환 파이프라인 (증강 없음)
        return A.Compose([
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ], bbox_params=BBOX_PARAMS)

#rotation(p=30, p=0.5)과 색상변환을 수행합니다. validation mAP: 0.8650
def get_A_transform7(train: bool):
    """
    Albumentations를 사용한 데이터 변환 파이프라인을 반환합니다.

    Args:
        train (bool): 훈련 모드인 경우 True, 아닐 경우 False.
    """
    if train:
        # 훈련용 데이터 증강 파이프라인
        return A.Compose([
            # --- 기하학적 변환 ---
            A.Rotate(limit=30, p=0.5, border_mode=cv2.BORDER_REPLICATE, value=0),

            # --- 색상 변환 ---
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),

            # --- 최종 변환 ---
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ], bbox_params=BBOX_PARAMS)
    else:
        # 검증/테스트용 변환 파이프라인 (증강 없음)
        return A.Compose([
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ], bbox_params=BBOX_PARAMS)


