# **경구 약제 이미지 검출 프로젝트 (Pill Detection Project)**

## **1. 프로젝트 개요 (Overview)**

*   **목표:** 이미지 속에 있는 최대 4개의 알약 종류(Class)와 위치(Bounding Box)를 검출하는 딥러닝 모델을 개발합니다.
*   **핵심 과제:** 시각적으로 유사한 수많은 종류의 알약들을 정확하게 구분하는 '세밀한 분류(Fine-grained Classification)' 문제를 해결합니다.
*   **평가 지표:** `mAP@0.5` (Mean Average Precision at IoU threshold 0.5)

## **2. 기술 스택 및 아키텍처 (Tech Stack & Architecture)**

*   **주요 라이브러리:** PyTorch, TorchVision, `torchmetrics`, `tqdm`, `uv`
*   **프로젝트 구조:** 재사용성과 확장성을 고려한 모듈형 구조 채택
    *   `configs`: 실험 설정을 관리하는 중앙 저장소
    *   `data`: `PillDataset` 및 데이터 변환 로직 포함
    *   `models`: Faster R-CNN, (향후 추가될) EfficientDet, DINO 등 모델 아키텍처
    *   `engine`: 훈련/평가/콜백 등 핵심 로직 포함
    *   `tools`: 훈련 및 평가를 위한 실행 스크립트
*   **개발 철학:**
    *   **TDD (Test-Driven Development):** `pytest`를 활용한 테스트 코드 작성을 통해 코드의 안정성 확보
    *   **재현성:** `pyproject.toml`과 `uv`를 통한 완벽한 의존성 버전 관리
    *   **모듈화:** 코드의 각 부분을 독립적으로 개발하고 테스트할 수 있도록 설계

## **3. 설치 및 환경 설정 (Installation & Setup)**

> 이 섹션은 새로운 팀원이나 사용자가 프로젝트를 시작하는 방법을 단계별로 안내합니다.

1.  **저장소 복제 (Clone Repository):**
    ```bash
    git clone [Your-Repository-URL]
    cd sprint-ai03-team04
    ```
2.  **가상 환경 생성 및 활성화 (Create & Activate Virtual Environment):**
    ```bash
    uv venv
    source .venv/bin/activate
    ```
3.  **의존성 설치 (Install Dependencies):**
    *   **1단계: 고정된 버전의 외부 라이브러리 설치 (재현성 보장)**
        ```bash
        uv pip install -r requirements.txt
        ```
    *   **2단계: 우리 프로젝트를 편집 가능 모드로 설치**
        ```bash
        uv pip install -e .
        ```

## **4. 사용 방법 (Usage)**

> 모델을 훈련하고 평가하는 구체적인 명령어를 제공합니다.
> 모든 스크립트는 프로젝트의 루트 디렉토리에서 실행하는 것을 기준으로 합니다.

1.  **모델 훈련 (Training):**
    *   `configs/base_config.py` 파일에서 배치 사이즈, 에폭 수 등 원하는 하이퍼파라미터를 수정합니다.
    *   아래 명령어를 실행하여 훈련을 시작합니다.
    ```bash
    python tools/train.py
    ```
    *   훈련 과정에서 최고 성능의 모델들은 `checkpoints/` 디렉토리에 자동으로 저장됩니다.

2.  **독립적인 평가 (Evaluation):**
    *   훈련된 모델 가중치 파일의 성능(mAP)을 측정하려면 아래 명령어를 실행합니다.
    *   `--checkpoint` 인자는 필수입니다.
    ```bash
    # 예시: checkpoints 폴더 안의 특정 모델 평가
    python tools/evaluate.py --checkpoint checkpoints/EPOCH\(12\)-LOSS\(0.0987\).pth
    ```

    *   만약 다른 설정 파일로 평가하고 싶다면 `--config` 인자를 사용하세요.
    ```bash
    # 예시: 다른 설정 파일로 평가
    python tools/evaluate.py --checkpoint [path/to/model] --config configs.another_config
    ```


3. **최종 제출 파일 생성 (Prediction for Submission):**

훈련된 모델을 사용하여 테스트 이미지에 대한 예측을 수행하고, 대회 제출 형식의 `submission.csv` 파일을 생성합니다.

   1.  **명령어 실행:**
       ```bash
       python tools/predict.py \
           --checkpoint [사용할 모델의 .pth 파일 경로] \
           --image_dir [테스트 이미지가 있는 폴더 경로] \
           --output [결과를 저장할 CSV 파일명 (기본값: submission.csv)]
       ```
   2.  **실행 예시:**
       ```bash
       python tools/predict.py \
           --checkpoint checkpoints/EPOCH\(12\)-LOSS\(0.0987\).pth \
           --image_dir path/to/your/test_images \
           --output my_submission.csv
       ```

## **5. 향후 개발 계획 (Future Work)**

> 이 프로젝트가 나아갈 방향을 제시하여 기여를 유도하고 로드맵을 공유합니다.

*   **[ ] 데이터 전처리 고도화:** `tools/preprocess_sam.py`를 구현하여 Segment Anything Model (SAM) 기반의 세그멘테이션 마스크 생성 및 Copy-Paste 데이터 증강 적용
*   **[ ] 모델 아키텍처 확장:**
    *   **EfficientDet:** 1-Stage Detector를 구현하여 베이스라인 성능과 비교
    *   **DINO:** Transformer 기반의 End-to-End 모델을 구현하여 SOTA 성능에 도전
*   **[ ] MLOps 도입:** `Weights & Biases` 또는 `MLflow`를 연동하여 실험 추적 및 결과 시각화 자동화
