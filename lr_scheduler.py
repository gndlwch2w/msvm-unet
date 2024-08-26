from __future__ import annotations
from torch.optim.lr_scheduler import *

LR_SCHEDULERS = {
    "PolynomialLR": PolynomialLR,
    "CosineAnnealingLR": CosineAnnealingLR,
    "CosineAnnealingWarmRestarts": CosineAnnealingWarmRestarts
}
