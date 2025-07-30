# tools/train.py
import torch
from torch.utils.data import DataLoader, Subset, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

import mlflow
import mlflow.pytorch # PyTorch 자동 로깅을 위해 임포트 (선택사항이지만 강력함)
import numpy as np # random seed 설정을 위해
import random
import os
import argparse
import importlib
from pprint import pprint

# 로컬 모듈 임포트
from data.dataset import PillDataset
from data.transforms import get_transform
from models.model_factory import build_model
from models.faster_rcnn import create_faster_rcnn_model
from utils.optimizer_factory import create_optimizer
from utils.scheduler_factory import create_scheduler

import configs._base_.models.faster_rcnn_r50_fpn
from utils.config_utils import recursive_update, DictAction, Config

from engine.trainer import train_one_epoch
from engine.evaluator import evaluate
from engine.callbacks import EarlyStopping, CheckpointSaver
from utils.data_utils import collate_fn

def main():
    # 1. 어떤 설정을 사용할지, 밖에서(명령줄에서) 주입받는다.
    parser = argparse.ArgumentParser(description="Pill Detection 모델을 훈련합니다.")
    parser.add_argument(
        '--config',
        required=True,
        help="사용할 설정 파일의 경로 (예: configs/faster_rcnn_resnet50.py)"
    )
    # 나중에 --options 와 같은 다른 인자들도 여기에 추가할 수 있습니다.
    parser.add_argument('--options', nargs='+', action=DictAction,
                        help='Override config options')
    parser.add_argument('--run_name', type=str, help='Custom run name for MLflow')
    args = parser.parse_args()

    # --- 2. 설정 로드 (레시피 해석) ---
    # 파싱된 인자(args.config)를 사용하여 필요한 설정 모듈을 로드합니다.
    try:
        # 문자열 경로를 Python 모듈 경로로 변환한다.
        # 'configs/my_experiment.py' -> 'configs.my_experiment'
        config_module_path = args.config.replace('/', '.').replace('.py', '')
        print(config_module_path)
        # 모듈을 동적으로 import 한다. 이것이 'load_config'의 핵심 원리.
        cfg_module = importlib.import_module(config_module_path)
        print(f"성공적으로 설정 파일을 로드했습니다: {args.config}")
    except ImportError:
        print(f"오류: 설정 파일 '{args.config}'를 찾을 수 없습니다.")
        return

    # _base_는 제외하고, 모든 변수(dict)를 가져옴
    cfg = {
        name: getattr(cfg_module, name) for name in dir(cfg_module)
        if not name.startswith('__') and name != '_base_'
    }

    if args.options:
        print("\n--- 명령줄 인자로 Config 덮어쓰기 ---")
        # cfg_dict를 args.options 딕셔너리로 재귀적으로 업데이트
        cfg = recursive_update(cfg, args.options)

    # 최종 딕셔너리를 점으로 접근 가능한 Namespace 객체로 변환
    cfg = Config.from_dict(cfg)

    pprint(cfg)

    # --- 재현성을 위한 시드(seed) 설정 ---
    # (MLflow로 실험을 추적할 때, 재현성은 매우 중요합니다)
    # seed = cfg.get('seed', 42) # config에 SEED가 없으면 기본값 42 사용
    seed = cfg.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    DEVICE = torch.device(cfg.device)
    print(f"사용할 장치: {DEVICE}")

    # --- MLflow 실험 설정 ---
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    # 데이터셋 준비
    dataset = PillDataset(root=cfg.data.root_dir, transforms=get_transform(train=True))
    dataset_val = PillDataset(root=cfg.data.root_dir, transforms=get_transform(train=False))

    num_classes = dataset.get_num_classes()

    # 훈련/검증 데이터셋 분할
    indices = torch.randperm(len(dataset)).tolist()
    train_size = int(len(dataset) * cfg.data.dataset_args.train_valid_split)
    train_subset = Subset(dataset, indices[:train_size])
    valid_subset = Subset(dataset_val, indices[train_size:])

    # 데이터로더 생성
    data_loader_train = DataLoader(train_subset, batch_size=cfg.data.train_batch_size, shuffle=True, num_workers=cfg.data.num_workers, collate_fn=collate_fn)
    data_loader_valid = DataLoader(valid_subset, batch_size=cfg.data.eval_batch_size, shuffle=False, num_workers=cfg.data.num_workers, collate_fn=collate_fn)

    print("데이터 준비 완료.")

    # --- MLflow 실행 시작 ---
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"\n--- MLflow Run 시작 (ID: {run_id}) ---")

         # --- 1. 파라미터 로깅 ---
        print("하이퍼파라미터 로깅...")
        # config 파일의 주요 설정들을 로깅
        params_to_log = {
            'seed': seed,
            'train_valid_split': cfg.data.dataset_args.train_valid_split,
            'num_classes': num_classes,
            'train_batch_size': cfg.data.train_batch_size,
            'eval_batch_size': cfg.data.eval_batch_size,
            'epochs': cfg.training.num_epochs,
            'learning_rate': cfg.optimizer.lr,
            'momentum': cfg.optimizer.momentum,
            'weight_decay': cfg.optimizer.weight_decay,
            'optimizer_type': cfg.optimizer.type,
            'es_patience': cfg.callbacks.early_stopping.patience,
            'min_box_size': cfg.data.dataset_args.min_box_size,
        }
        mlflow.log_params(params_to_log)
        # config 파일 자체를 아티팩트로 저장하는 것도 좋은 방법입니다.
        if os.path.exists(args.config):
            mlflow.log_artifact(args.config, artifact_path="configs")

        # --- 2. 모델 및 훈련 도구 준비 ---
        print("모델 및 훈련 도구 준비 중...")

        # 설정 파일에는 num_classes 정보가 없을 수 있으므로, 동적으로 추가
        cfg.model['num_classes'] = num_classes

        # 팩토리를 통해 모델 생성
        model = build_model(cfg.model).to(DEVICE)
        optimizer = create_optimizer(model, cfg.optimizer)
        lr_scheduler = create_scheduler(optimizer, cfg.lr_scheduler)

        # 모델, 옵티마이저, 콜백 준비
        # model = create_faster_rcnn_model(num_classes).to(cfg.DEVICE)
        # optimizer = torch.optim.SGD(model.parameters(), lr=cfg.LEARNING_RATE, momentum=cfg.MOMENTUM, weight_decay=cfg.WEIGHT_DECAY)
        # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=cfg.lr_patience, verbose=True, min_lr=1e-6)

        early_stopper = EarlyStopping(patience=cfg.callbacks.early_stopping.patience, verbose=True)
        checkpoint_saver = CheckpointSaver(save_dir=cfg.callbacks.checkpoint_saver.save_dir, top_k=cfg.callbacks.checkpoint_saver.top_k, verbose=True)

        best_known_val_loss = np.inf # 최고 손실 기록을 위한 변수

        # 훈련/검증 루프 시작
        for epoch in range(cfg.training.num_epochs):
            # --- 훈련 ---
            model.train() # 훈련 시작 전, 상태를 명시적으로 설정
            avg_train_loss = train_one_epoch(model, optimizer, data_loader_train, DEVICE, epoch, cfg.training.num_epochs)
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)

            # 주기에 따라 mAP 계산 여부 결정
            is_map_cycle = (epoch + 1) % cfg.training.map_calc_cycle == 0 or (epoch + 1) == cfg.training.num_epochs

            # --- 검증 ---
            model.eval() # 검증/평가 시작 전, 상태를 명시적으로 설정
            avg_val_loss, map_results = evaluate(
                model, data_loader_valid,
                DEVICE,
                calculate_map_metric=is_map_cycle # 플래그 전달
            )

            # 검증 지표 로깅
            val_metrics = {'val_loss': avg_val_loss}
            if map_results:
                val_metrics['val_mAP_50'] = map_results['map_50'].item()
            mlflow.log_metrics(val_metrics, step=epoch)

            # # --- MLflow에 검증 지표 기록 ---
            # # 딕셔너리로 묶어서 한 번에 로깅
            # val_metrics = {
            #     'val_loss': avg_loss,
            #     'val_mAP': map_results['map'].item(),
            #     'val_mAP_50': map_results['map_50'].item(),
            #     'val_mAP_75': map_results['map_75'].item()
            # }
            # mlflow.log_metrics(val_metrics, step=epoch)
            # # ---

            # mAP는 매 에폭마다 계산하면 시간이 오래 걸리므로,
            # 마지막 에폭이나 특정 주기로만 계산할 수 있습니다.
            # if (epoch + 1) % 5 == 0 or epoch == cfg.NUM_EPOCHS - 1:
            #     map_results = calculate_map(...)

            # 스케줄러 타입에 따라 step() 호출 방식이 다름
            if lr_scheduler:
                # ReduceLROnPlateau는 검증 손실을 필요인자로 받음
                if isinstance(lr_scheduler, ReduceLROnPlateau):
                    lr_scheduler.step(avg_val_loss)
                # 다른 스케줄러들은 에폭이 끝날 때 호출
                else:
                    lr_scheduler.step()

            # 로그 출력부 수정
            log_message = f"Epoch {epoch+1}: TrainLoss={avg_train_loss:.4f}, ValidLoss={avg_val_loss:.4f}"
            if map_results: # map_results가 None이 아닐 때만 출력
                log_message += f", mAP@0.5={map_results['map_50']:.4f}"
            print(log_message)

            # --- Refined Callback and Artifact Logging Logic ---
            # 1. CheckpointSaver simply saves files based on its Top-K logic.
            checkpoint_saver(avg_val_loss, epoch, model)

            # 2. The main loop checks if the best score was beaten.
            if avg_val_loss < best_known_val_loss:
                print(f"  -> ⭐ 최고 성능 갱신! 검증 손실: ({best_known_val_loss:.4f} -> {avg_val_loss:.4f})")
                best_known_val_loss = avg_val_loss

                # 고유한 경로에 저장
                artifact_sub_path = f"checkpoints/epoch_{epoch+1}"
                mlflow.log_artifact(best_model_path, artifact_path=artifact_sub_path)

            if early_stopper(avg_val_loss):
                print("EarlyStopping 조건 충족. 훈련을 조기 종료합니다.")
                break

        print("\n훈련 종료. 최종 저장된 모델 목록:")
        for loss, path in checkpoint_saver.checkpoints:
            print(f"  - {path} (검증 손실: {loss:.4f})")

        print("\n최종 모델을 MLflow에 등록합니다.")
        # Find the corresponding checkpoint file just saved
        # The best model is always the first in the sorted list.
        if checkpoint_saver.checkpoints:
            best_model_path = checkpoint_saver.checkpoints[0][1]
            print(f"  -> 새로운 최고 모델을 MLflow에 아티팩트로 기록: {best_model_path}")

            # 최종 모델 객체의 상태를 최고 성능 가중치로 업데이트합니다.
            model.load_state_dict(torch.load(best_model_path))

            # mlflow.pytorch.log_model을 사용하여 기록과 등록을 동시에 수행합니다.
            mlflow.pytorch.log_model(
                pytorch_model=model, # 최종 모델 객체
                artifact_path="final_model",
                registered_model_name="Pill_Detector_FasterRCNN"
            )
            print("최종 모델이 MLflow 아티팩트 및 모델 레지스트리에 성공적으로 저장/등록되었습니다.")

    print(f"--- MLflow Run 종료 (ID: {run_id}) ---")
    print(f"MLflow UI를 실행하여 결과를 확인하세요: mlflow ui --backend-store-uri {cfg.mlflow.tracking_uri}")


if __name__ == '__main__':
    main()
