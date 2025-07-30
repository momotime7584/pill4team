# configs/_base_/datasets/pill_detection.py
# 데이터셋 관련 설정만 정의

data = dict(
    root_dir='<your-own-path>',
    train_path='train_images',
    ann_path='train_annotations',

    # 데이터 로더 설정
    num_workers=2,
    train_batch_size=8,
    eval_batch_size=16,
    
    # Dataset 클래스에 전달될 인자들 설정
    dataset_args=dict(
        min_box_size=10,
        train_valid_split=0.8
    ),
    
    # Augmentation 파이프라인 설정
    # 데이터 증강 설정 (나중에 transforms.py에서 이 dict를 참조)
    augmentation=dict(
        img_size=512,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
        # ... 기타 albumentations 파라미터 ...
    )
)
