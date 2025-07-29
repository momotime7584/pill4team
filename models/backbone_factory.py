# models/backbone_factory.py
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
# 나중에 SwinTransformer 등을 추가하려면 여기에 import

def build_backbone(config):
    """
    설정 딕셔너리를 바탕으로 백본 객체를 생성합니다.
    """
    config_copy = config.copy()

    backbone_type = config_copy.pop('type') # config에서 type을 꺼냄
    backbone_name = config_copy.pop('name') # 'name'을 꺼내서 변수에 저장

    if backbone_type == 'ResNetFPN':
        # config의 나머지 내용(depth, pretrained 등)을 인자로 전달
        return resnet_fpn_backbone(backbone_name=backbone_name, **config_copy)
    # elif backbone_type == 'SwinTransformer':
    #     return build_swin_transformer_backbone(**config)
    else:
        raise ValueError(f"Unknown backbone type: {backbone_type}")
