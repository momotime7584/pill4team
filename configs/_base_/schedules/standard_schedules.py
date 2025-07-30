# configs/_base_/schedules/standard_schedules.py

# --- Optimizer Presets ---

sgd_optimizer = dict(
    type='SGD', 
    lr=0.005, 
    momentum=0.9, 
    weight_decay=0.0005
)

adamw_optimizer = dict(
    type='AdamW',
    lr=0.0001,
    weight_decay=0.01
)

# --- LR Scheduler Presets ---

steplr_scheduler_12ep = dict(
    type='StepLR',
    step_size=8, # 12 에폭 훈련 시 8, 11 에폭에서 LR 감소
    gamma=0.1
)

reduce_lr_on_plateau_scheduler = dict(
    type='ReduceLROnPlateau',
    mode='min',
    factor=0.1,
    patience=3 # LR_PATIENCE
)

# --- Training Length Presets ---

train_12ep = dict(num_epochs=12)
train_24ep = dict(num_epochs=24)