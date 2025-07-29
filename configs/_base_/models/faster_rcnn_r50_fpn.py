# configs/_base_/models/faster_rcnn_r50_fpn.py

model = dict(
    type='FasterRCNN',
    backbone=dict(
        type='ResNetFPN',
        name='resnet50',
        pretrained=True
    ),
    # RPN (Region Proposal Network) 설정
    rpn=dict(
        anchor_sizes=((32,), (64,), (128,), (256,), (512,)),
        aspect_ratios=((0.5, 1.0, 2.0),)
    ),
    # RoI (Region of Interest) Head 설정
    roi_head=dict(
        # 훈련 중 NMS 전/후에 유지할 제안 수
        pre_nms_top_n_train=2000,
        post_nms_top_n_train=2000,
        # 테스트 중 NMS 전/후에 유지할 제안 수
        pre_nms_top_n_test=1000,
        post_nms_top_n_test=1000,
        rpn_nms_thresh=0.7, # RPN NMS 임계값

        # 최종 예측에 대한 임계값
        score_thresh=0.05, # ✅ 이 값이 바로 box_score_thresh 입니다.
        box_nms_thresh=0.5,
        detections_per_img=100,
    )
)