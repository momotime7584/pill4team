# engine/trainer.py
import torch
from tqdm.auto import tqdm

def train_one_epoch(model, optimizer, data_loader, device, epoch, num_epochs, grad_clip_norm=0.0):
    """한 에폭의 훈련을 수행합니다."""
    total_loss = 0.0
    
    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs} [훈련]")
    for images, targets in progress_bar:
        if images is None: continue

        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()

        # 그래디언트 클리핑 (값이 0보다 클 때만 실행)
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

        optimizer.step()

        total_loss += losses.item()
        progress_bar.set_postfix(loss=losses.item())
        
    return total_loss / len(data_loader)

def train(model, optimizer, data_loader, device, num_epochs, scheduler=None, val_data_loader=None):
    """
    모델 훈련을 시작하고 각 에폭마다 훈련 및 검증을 수행합니다.
    스케줄러가 제공되면 검증 손실에 따라 학습률을 조정합니다.
    """
    for epoch in range(num_epochs):
        model.train()
        train_loss = train_one_epoch(model, optimizer, data_loader, device, epoch, num_epochs)
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}")

        if scheduler and val_data_loader:
            # 검증 데이터 로더를 사용하여 검증 손실 계산 (여기서는 더미 값 사용)
            # 실제 구현에서는 evaluator.py의 evaluate 함수를 호출하여 검증 손실을 얻어야 합니다.
            val_loss = train_loss # 임시로 훈련 손실을 검증 손실로 사용. 실제로는 evaluate 함수 호출 필요.
            scheduler.step(val_loss)
