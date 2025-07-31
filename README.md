# **경구 약제 이미지 검출 프로젝트 (Pill Detection Project)**

## 0. 발표 자료
[Notion presentatoin](https://purring-pullover-e72.notion.site/Presentation-240e67a1ff23801bb455e4b8deac0c5e?source=copy_link)


## **1. 프로젝트 개요 (Overview)**

본 프로젝트는 이미지 속에 있는 최대 4개의 알약 종류(Class)와 위치(Bounding Box)를 검출하는 딥러닝 모델 개발을 목표로 합니다. 시각적으로 유사한 수많은 알약들을 정확하게 구분하는 Object Detection 문제를 해결하고, 체계적인 실험 관리를 통해 최적의 모델을 찾아냅니다.

*   **핵심 과제:** Fine-grained Object Detection
*   **평가 지표:** `mAP@.50` (Mean Average Precision at IoU threshold 0.5)

## **2. 아키텍처 및 기술 스택 (Architecture & Tech Stack)**

*   **주요 라이브러리:** PyTorch, Albumentations, MLflow, DagsHub, `uv`, `pytest`
*   **핵심 아키텍처:**
    *   **Configuration System:** Python(`.py`) 기반의 'Lazy Config' 시스템을 채택하여, 재사용 가능한 설정('`_base_`')과 개별 실험 설정을 명확히 분리합니다.
    *   **Factory Pattern:** `build_model`, `create_optimizer` 등 팩토리 함수를 통해 Config 파일과 실제 구현을 분리하여 확장성을 극대화합니다.
    *   **MLOps:** DagsHub를 중앙 서버로 사용하는 MLflow를 통해 모든 실험의 파라미터, 지표, 아티팩트를 추적하고 팀원 간 공유합니다.
*   **프로젝트 구조:**
    *   `configs/`: 모든 실험의 '설계도'가 위치하는 곳.
    *   `data/`, `models/`, `engine/`, `utils/`: 데이터, 모델, 훈련 엔진 등 재사용 가능한 핵심 로직 모듈.
    *   `tools/`: `train.py`, `evaluate.py` 등 안정적인 핵심 실행 '도구'.
    *   `experiments/`: 하이퍼파라미터 튜닝 등 다양한 실험을 자동화하는 '실행 스크립트'.
    *   `tests/`: `pytest`를 사용한 단위 테스트 코드.

## **3. 시작하기 (Getting Started)**

### **3.1. 최초 환경 설정 (First-Time Setup)**

1.  **저장소 복제:**
    ```bash
    git clone https://github.com/momotime7584/pill4team
    cd pill4team
    ```
2.  **DagsHub 인증:**
    *   [DagsHub 프로젝트](https://dagshub.com/jehakim2210/codeit-project1-team4)에 Collaborator로 참여합니다.
    *   `pip install dagshub`를 실행합니다.
    *   Python 스크립트나 노트북에서 `import dagshub;` 후에 실행하여 최초 1회 인증을 완료합니다.

3.  **가상 환경 생성 및 의존성 설치:**
    ```bash
    # 가상 환경 생성 및 활성화
    uv venv
    source .venv/bin/activate

    # 의존성 잠금 파일 생성 (최초 1회 또는 의존성 변경 시)
    uv pip compile requirements/cuda.in -o requirements.txt # CUDA 환경
    # uv pip compile requirements/macos.in -o requirements.txt # macOS 환경

    # 의존성 설치
    uv pip install -r requirements.txt
    uv pip install -e .
    ```

### **3.2. GPU 환경 확인 (NVIDIA GPU)**

`requirements/cuda.in` 파일은 CUDA 12.1 (`cu121`)을 기준으로 작성되었습니다. `nvidia-smi`를 통해 시스템의 CUDA 버전을 확인하고, 필요 시 `.in` 파일의 `--extra-index-url`을 수정하세요.

---

## **4. 사용 방법 (Usage)**

### **4.1. 단일 실험 실행 (Single Experiment)**

특정 실험 설계를 실행합니다. `configs/` 폴더에 있는 `.py` 파일을 수정하거나 새로 만들어 실험을 정의할 수 있습니다.

```bash
python tools/train.py configs/faster_rcnn_r50_fpn_baseline.py
```

### **4.2. 자동화된 하이퍼파라미터 탐색 (Hyperparameter Tuning)**

`experiments/` 폴더의 스크립트를 사용하여 여러 실험을 자동으로 실행합니다. 스크립트 내부의 `param_grid`를 수정하여 탐색할 파라미터 조합을 정의할 수 있습니다.

```bash
python experiments/01_broad_lr_search.py
```

### **4.3. 모델 평가 (Evaluation)**

훈련된 모델(`.pth`) 파일의 성능(mAP)을 측정합니다.

```bash
python tools/evaluate.py --checkpoint checkpoints/EPOCH-LOSS-best.pth
```

### **4.4. 최종 제출 파일 생성 (Prediction)**

테스트 이미지에 대한 예측을 수행하고, `submission.csv` 파일을 생성합니다.

```bash
python tools/predict.py \
    --checkpoint checkpoints/EPOCH-LOSS-best.pth \
    --image_dir path/to/test_images
```


### **4.5. 결과 재현 및 브랜치 안내 (Reproducing Results & Branch Guide)**

이 프로젝트는 개발 목적에 따라 여러 브랜치로 관리되고 있습니다. 각 브랜치의 역할과 최고 성능을 재현하는 방법은 다음과 같습니다.

*   **`main` 브랜치 (안정 버전 / Stable):**
    *   **설명:** MLflow, DagsHub, 자동화된 실험 등 완전한 MLOps 파이프라인이 통합된 가장 안정적인 버전입니다. 모든 새로운 기능 개발은 이 브랜치를 기준으로 진행됩니다.

*   **`performance-test` 브랜치 (최고 성능 버전 / Highest Performance):**
    *   **설명:** 데이터 전처리 및 증강 실험에 집중하여 현재까지 가장 높은 성능을 달성한 개발 브랜치입니다.
    *   **최고 스코어:** **`mAP@.50 = 0.99451`**
    *   **재현 방법:** 최고 성능을 직접 재현하려면, 이 브랜치로 전환하여 훈련을 실행해야 합니다.
        ```bash
        # 1. performance-test 브랜치로 체크아웃
        git switch performance-test

        # 2. 해당 브랜치의 가이드에 따라 훈련 실행
        # (예시 명령어이며, 해당 브랜치의 configs 폴더를 참고하세요)
        python tools/train.py
        ```

**향후 계획:** 현재 `performance-test` 브랜치에서 검증된 핵심적인 데이터 처리 및 모델링 기법들을 `main` 브랜치의 MLOps 파이프라인으로 점진적으로 통합하여, 최고 성능을 안정적으로 재현하고 추가적인 개선을 진행할 예정입니다.


---

## **5. MLOps: 실험 추적 및 공유**

모든 훈련 실행은 DagsHub의 중앙 MLflow 서버에 자동으로 기록됩니다.

1.  **실행:** `tools/train.py` 또는 `experiments/` 스크립트를 실행합니다.
2.  **확인:** [DagsHub Experiments](https://dagshub.com/jehakim2210/codeit-project1-team4/experiments/) 또는 [MLflow UI](https://dagshub.com/jehakim2210/codeit-project1-team4.mlflow)에서 모든 팀원의 실험 결과를 실시간으로 확인하고 비교 분석할 수 있습니다.

---

## **6. 향후 개발 계획 (Future Work)**

*   **[ ] 데이터 전처리 고도화:** SAM(Segment Anything Model)을 활용한 `tools/preprocess_sam.py`를 구현하여 Copy-Paste 증강 적용.
*   **[ ] 모델 아키텍처 확장:** DETR, DINO 등 최신 아키텍처를 `models/`에 추가.
*   **[ ] 자동화된 튜닝:** `Optuna`를 `experiments/` 스크립트에 통합하여 더 지능적인 하이퍼파라미터 탐색 수행.
