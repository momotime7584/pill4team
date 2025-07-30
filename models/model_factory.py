# models/model_factory.py
from .backbone_factory import build_backbone
from .faster_rcnn import build_faster_rcnn_model
# from .dino import build_dino_model # 나중에 추가

def build_model(config):
    """
    전체 모델 설정을 받아, 각 부품을 만들고 최종 모델을 조립합니다.
    """
    config_copy = config.copy()
    
    # 1. 백본 설정을 분리하여 백본 팩토리에 전달
    backbone_config = config_copy.pop('backbone')
    backbone = build_backbone(backbone_config)

    # 2. 모델 타입을 기준으로 분기
    model_type = config_copy.pop('type')
    if model_type == 'FasterRCNN':
        # 3. 조립된 백본과 나머지 모델 설정들을 모델 생성 함수에 전달
        return build_faster_rcnn_model(backbone=backbone, **config_copy)
    # elif model_type == 'DINO':
    #     return build_dino_model(backbone=backbone, **config_copy)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
