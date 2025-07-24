# tools/train.py
import torch
from torch.utils.data import DataLoader, Subset, random_split

# 로컬 모듈 임포트
from configs import base_config as cfg
from data.dataset import PillDataset
from data.transforms import get_transform
from models import create_model
from engine.trainer import train_one_epoch
from engine.evaluator import evaluate
from engine.callbacks import EarlyStopping, CheckpointSaver

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch: return (None, None)
    return tuple(zip(*batch))


def main():
    print(f"사용할 장치: {cfg.DEVICE}")

    # 데이터셋 준비
    dataset = PillDataset(
        root=cfg.ROOT_DIRECTORY, 
        transforms=get_transform(train=True),
        min_box_size=cfg.MIN_BOX_SIZE)
    dataset_val = PillDataset(
        root=cfg.ROOT_DIRECTORY, 
        transforms=get_transform(train=False),
        min_box_size=cfg.MIN_BOX_SIZE)

    num_classes = len(dataset.map_cat_id_to_label) + 1

    # 훈련/검증 데이터셋 분할
    indices = torch.randperm(len(dataset)).tolist()
    train_size = int(len(dataset) * cfg.TRAIN_VALID_SPLIT)
    train_subset = Subset(dataset, indices[:train_size])
    valid_subset = Subset(dataset_val, indices[train_size:])

    # 데이터로더 생성
    data_loader_train = DataLoader(train_subset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn)
    data_loader_valid = DataLoader(valid_subset, batch_size=cfg.BATCH_SIZE_VAL, shuffle=False, num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn)

    # 모델, 옵티마이저, 콜백 준비
    model = create_model("faster_rcnn", num_classes).to(cfg.DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.LEARNING_RATE, momentum=cfg.MOMENTUM, weight_decay=cfg.WEIGHT_DECAY)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    early_stopper = EarlyStopping(patience=cfg.ES_PATIENCE, verbose=True)
    checkpoint_saver = CheckpointSaver(save_dir=cfg.CHECKPOINT_DIR, top_k=cfg.CS_TOP_K, verbose=True)

    # 훈련/검증 루프 시작
    for epoch in range(cfg.NUM_EPOCHS):
        # --- 훈련 ---
        model.train() # 훈련 시작 전, 상태를 명시적으로 설정
        avg_train_loss = train_one_epoch(
            model, optimizer, data_loader_train, 
            cfg.DEVICE, epoch, cfg.NUM_EPOCHS,
            grad_clip_norm=cfg.GRADIENT_CLIP_NORM)

        # 주기에 따라 mAP 계산 여부 결정
        is_map_cycle = (epoch + 1) % cfg.MAP_CALC_CYCLE == 0 or (epoch + 1) == cfg.NUM_EPOCHS
    
        # --- 검증 ---
        model.eval() # 검증/평가 시작 전, 상태를 명시적으로 설정
        avg_val_loss, map_results = evaluate(
            model, data_loader_valid, 
            cfg.DEVICE,
            calculate_map_metric=is_map_cycle # 플래그 전달
        )

        # mAP는 매 에폭마다 계산하면 시간이 오래 걸리므로,
        # 마지막 에폭이나 특정 주기로만 계산할 수 있습니다.
        # if (epoch + 1) % 5 == 0 or epoch == cfg.NUM_EPOCHS - 1:
        #     map_results = calculate_map(...)

        lr_scheduler.step()

        # 로그 출력부 수정
        log_message = f"Epoch {epoch+1}: TrainLoss={avg_train_loss:.4f}, ValidLoss={avg_val_loss:.4f}"
        if map_results: # map_results가 None이 아닐 때만 출력
            log_message += f", mAP@0.5={map_results['map_50']:.4f}"
        print(log_message)

        checkpoint_saver(avg_val_loss, epoch, model)
        early_stopper(avg_val_loss)
        if early_stopper.early_stop:
            break

    print("\n훈련 종료. 최종 저장된 모델 목록:")
    for loss, path in checkpoint_saver.checkpoints:
        print(f"  - {path} (검증 손실: {loss:.4f})")

if __name__ == '__main__':
    main()