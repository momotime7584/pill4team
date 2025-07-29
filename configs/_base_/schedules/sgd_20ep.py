# configs/_base_/schedules/sgd_20ep.py

# optimizer = dict(type='SGD', lr=0.005, momentum=0.9)
# ... 기타 스케줄러, 에폭 설정

# 옵티마이저 (기본값: SGD)
optimizer = dict(
    type='SGD', 
    lr=0.005, 
    momentum=0.9, 
    weight_decay=0.0005
)

# 학습률 스케줄러
lr_scheduler = dict(
    type='StepLR', 
    step_size=8, 
    gamma=0.1
)
