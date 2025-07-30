# 1. 부품들을 import
from ._base_.models.faster_rcnn_r50_fpn import model
from ._base_.datasets.pill_detection import data
from ._base_.schedules.sgd_20ep import optimizer, lr_scheduler
from ._base_.default_runtime import seed, device, callbacks, mlflow, training

# 2. 이번 실험에서 바꿀 것만 덮어쓰기
optimizer.update(lr=0.01)
data['train_batch_size'] = 16
