# data/transforms.py
import torch
import torchvision.transforms.v2 as T

# "하드코딩 된 이미지 정규화 값" base_config.py에서 가져오는 것이 좋습니다.
# 하지만 여기서는 예시로 하드코딩된 값을 사용합니다.
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

def get_basic_transform():
    transforms = []
    transforms.append(T.ToImage())
    transforms.append(T.ToDtype(torch.float32, scale=True))
    transforms.append(T.Normalize(mean=MEAN, std=STD)) # 정규화 추가
    return T.Compose(transforms)


def get_transform(train: bool):
    transforms = []
    transforms.append(T.ToImage())
    transforms.append(T.ToDtype(torch.float32, scale=True))
    transforms.append(T.Normalize(mean=MEAN, std=STD))#정규화 추가
    if train:
        transforms.append(T.RandomVerticalFlip(0.5)) # 수직 뒤집기로 변경
        
    return T.Compose(transforms)


def get_augmented_transform():
    # 추후 이 곳에 회전, 밝기 조절, CopyPaste 등의 증강 로직을 추가합니다.
    pass

