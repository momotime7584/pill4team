# utils/scheduler_factory.py
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR

def create_scheduler(optimizer, config):
    """
    설정 딕셔너리를 바탕으로 학습률 스케줄러 객체를 생성합니다.
    """
    # config 딕셔너리에서 scheduler 설정 부분만 추출
    # cfg.lr_scheduler 와 같은 객체가 전달됨
    scheduler_config = config 
    
    # 설정이 없거나 None이면 스케줄러를 사용하지 않음
    if not scheduler_config:
        return None

    scheduler_type = scheduler_config.get('type', 'StepLR').lower()

    if scheduler_type == 'steplr':
        return StepLR(
            optimizer,
            step_size=scheduler_config['step_size'],
            gamma=scheduler_config.get('gamma', 0.1)
        )
    elif scheduler_type == 'reducelronplateau':
        return ReduceLROnPlateau(
            optimizer,
            mode=scheduler_config.get('mode', 'min'),
            factor=scheduler_config.get('factor', 0.1),
            patience=scheduler_config.get('patience', 10),
            verbose=True
        )
    elif scheduler_type == 'cosineannealinglr':
        # Cosine 스케줄러는 총 에폭 수 또는 스텝 수가 필요
        if 'T_max' not in scheduler_config:
            raise ValueError("CosineAnnealingLR requires 'T_max' in its config.")
        return CosineAnnealingLR(
            optimizer,
            T_max=scheduler_config['T_max']
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")