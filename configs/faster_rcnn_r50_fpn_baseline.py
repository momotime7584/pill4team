# configs/faster_rcnn_r50_fpn_baseline.py

from ._base_.models.faster_rcnn_r50_fpn import model
from ._base_.datasets.pill_detection import data
from ._base_.schedules.sgd_20ep import optimizer, lr_scheduler
from ._base_.default_runtime import seed, device, callbacks, mlflow, training

# 이 파일은 기본 부품들을 조립만 할 뿐, 아무것도 덮어쓰지 않습니다.
# 이것이 우리의 '바닐라' 모델입니다.