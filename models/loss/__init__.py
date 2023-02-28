from loss.loss import ComputeLoss, ComputeNanoLoss
from .ssod.ssod_loss import ComputeStudentMatchLoss


def build_loss(model, cfg):
    if cfg.Loss.type == 'ComputeLoss': 
        return ComputeLoss(model, cfg)  # init loss class
    else:
        raise NotImplementedError

def build_ssod_loss(model, cfg):
    if cfg.SSOD.loss_type == 'ComputeStudentMatchLoss':
        return ComputeStudentMatchLoss(model, cfg)
    else:
        raise NotImplementedError