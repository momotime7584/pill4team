# engine/callbacks.py
import torch
import os
import numpy as np

# =================================================================================
# 0. 콜백 클래스 정의 (EarlyStopping)
# =================================================================================
# 이 클래스는 훈련을 효율적으로 관리하는 전문가용 기능입니다.
class EarlyStopping:
    """검증 손실이 개선되지 않으면 훈련을 조기에 중단시킵니다."""
    def __init__(self, patience=5, min_delta=0, verbose=False):
        """
        Args:
            patience (int): 개선이 없다고 판단하기까지 기다릴 에폭 수.
            min_delta (float): 개선으로 인정할 최소 변화량.
            verbose (bool): 조기 중단 시 메시지를 출력할지 여부.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = np.inf

    def __call__(self, val_loss):
        if self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False # 개선되었으므로 중단 안 함
        else:
            self.counter += 1
            if self.counter >= self.patience:
                if self.verbose:
                    print(f"EarlyStopping: 검증 손실이 {self.patience} 에폭 동안 개선되지 않았습니다. 훈련을 중단합니다.")
                return True
        return False

# 0. 콜백 클래스 정의 (CheckpointSaver 추가)
class CheckpointSaver:
    """검증 손실을 기준으로 상위 K개의 모델만 저장하고 관리합니다."""
    def __init__(self, save_dir='checkpoints', top_k=3, verbose=False):
        """
        Args:
            save_dir (str): 체크포인트를 저장할 디렉토리.
            top_k (int): 유지할 상위 모델의 개수.
            verbose (bool): 모델 저장/삭제 시 메시지를 출력할지 여부.
        """
        self.save_dir = save_dir
        self.top_k = top_k
        self.verbose = verbose

        # (loss, filepath) 튜플을 저장할 리스트
        self.checkpoints = []
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def __call__(self, val_loss, epoch, model):
        # 파일명 생성 (Keras 스타일)
        filename = f"EPOCH({epoch+1:02d})-LOSS({val_loss:.4f}).pth"
        filepath = os.path.join(self.save_dir, filename)

        # 현재 저장된 체크포인트가 K개 미만이거나,
        # 현재 손실이 저장된 체크포인트 중 가장 나쁜(가장 큰) 손실보다 좋을 때만 저장
        if len(self.checkpoints) < self.top_k or val_loss < self.checkpoints[-1][0]:
            # 모델 저장
            torch.save(model.state_dict(), filepath)

            # 리스트에 추가
            self.checkpoints.append((val_loss, filepath))

            # 손실을 기준으로 오름차순 정렬 (가장 좋은 모델이 맨 앞에 오도록)
            self.checkpoints.sort(key=lambda x: x[0])

            if self.verbose:
                print(f"  -> 체크포인트 저장: {filepath} (검증 손실: {val_loss:.4f})")

            # # --- 최고 성능 모델이 갱신될 때만 MLflow에 아티팩트로 기록 ---
            # if self.run_id and (len(self.checkpoints) == 1 or val_loss == self.checkpoints[0][0]):
            #     print(f"  -> 최고 성능 모델을 MLflow에 아티팩트로 기록: {filepath}")
            #     mlflow.log_artifact(filepath, artifact_path="best_model")
            # # ---

            # 만약 저장된 체크포인트가 K개를 초과하면, 가장 나쁜 모델 삭제
            if len(self.checkpoints) > self.top_k:
                worst_checkpoint = self.checkpoints.pop() # 가장 마지막 요소 (가장 나쁜 손실)
                try:
                    os.remove(worst_checkpoint[1])
                    if self.verbose:
                        print(f"  -> 오래된 체크포인트 삭제: {worst_checkpoint[1]}")
                except OSError as e:
                    print(f"Error removing file {worst_checkpoint[1]}: {e}")
