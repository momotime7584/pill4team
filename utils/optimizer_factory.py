# utils/optimizer_factory.py
import torch.optim as optim

def create_optimizer(model, config):
    params = [p for p in model.parameters() if p.requires_grad]
    opt_config = config
    opt_type = opt_config.get('type', 'SGD').lower()

    if opt_type == 'sgd':
        return optim.SGD(
            params,
            lr=opt_config['lr'],
            momentum=opt_config.get('momentum', 0.9),
            weight_decay=opt_config.get('weight_decay', 0.0)
        )
    elif opt_type == 'adam':
        return optim.Adam(
            params,
            lr=opt_config['lr'],
            weight_decay=opt_config.get('weight_decay', 0.0)
        )
    elif opt_type == 'adamw':
        return optim.AdamW(
            params,
            lr=opt_config['lr'],
            weight_decay=opt_config.get('weight_decay', 0.01)
        )
    else:
        raise ValueError(f"Unknown optimizer type: {opt_type}")