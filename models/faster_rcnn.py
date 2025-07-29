# models/faster_rcnn.py
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision

def create_faster_rcnn_model(num_classes, backbone='resnet50', pretrained=True, **kwargs):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# models/faster_rcnn.py
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# 이 함수의 이름은 create_... 보다 build_... 가 팩토리 패턴과 더 어울림
def build_faster_rcnn_model(backbone, num_classes, **kwargs):
    """
    주입된 백본과 설정값들로 Faster R-CNN 모델을 조립합니다.
    """
    # 앵커 생성기 설정 (config에서 값을 받아옴)
    anchor_sizes = kwargs.get('anchor_sizes', ((32,), (64,), (128,), (256,), (512,)))
    aspect_ratios = kwargs.get('aspect_ratios', ((0.5, 1.0, 2.0),)) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)

    # RoI 풀러 설정 (필요 시 kwargs에서 받아옴)
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'],
                                                    output_size=7,
                                                    sampling_ratio=2)

    # 최종 모델 조립
    model = FasterRCNN(
        backbone,
        num_classes=91, # COCO의 기본 클래스 수로 초기화
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        **kwargs # rpn_pre_nms_top_n 등 나머지 파라미터 전달
    )
    
    # 분류기 헤드를 우리 데이터셋의 클래스 수에 맞게 교체
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model