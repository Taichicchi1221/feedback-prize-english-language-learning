import numpy as np
from sklearn.metrics import mean_squared_error

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

# ====================================================
# loss
# ====================================================
def get_loss(loss_type, loss_params):
    if loss_params is None:
        loss_params = {}

    if loss_type in ("MCRMSELoss", "SmoothL1Loss"):
        return eval(loss_type)(**loss_params)

    return getattr(torch.nn, loss_type)(**loss_params)


class MCRMSELoss(nn.Module):
    EPS = 1.0e-09

    def __init__(self) -> None:
        super().__init__()

    def forward(self, preds, target):
        return torch.mean(torch.sqrt(torch.mean(F.mse_loss(preds, target, reduction="none"), dim=0) + self.EPS), dim=0)


class SmoothL1Loss(nn.Module):
    def __init__(self, beta=1.0) -> None:
        super().__init__()
        self.beta = beta

    def forward(self, preds, target):
        return torch.mean(torch.mean(F.smooth_l1_loss(preds, target, reduction="none"), dim=0), dim=0)


# ====================================================
# metric
# ====================================================
def MCRMSE(y_true, y_preds):
    assert y_true.shape == y_preds.shape
    scores = np.sqrt(np.mean(np.square(y_true - y_preds), axis=0))
    mcrmse_score = np.mean(scores)
    return mcrmse_score, scores


class MCRMSEMetric(torchmetrics.Metric):
    full_state_update = False

    def __init__(self, num_cols):
        super().__init__()
        self.add_state("sum_values", default=torch.tensor([0] * num_cols, dtype=torch.float), dist_reduce_fx="sum")
        self.add_state("length", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, preds, target):
        assert preds.size() == target.size()
        self.sum_values += torch.sum(torch.square(preds.float() - target.float()), dim=0)
        self.length += target.size(0)

    def compute(self):
        return torch.mean(torch.sqrt(self.sum_values.float() / self.length.float()))


def get_metric(metric_type, metric_params):
    if metric_params is None:
        metric_params = {}

    if metric_type == "MCRMSEMetric":
        return eval(metric_type)(**metric_params)
    return getattr(torchmetrics, metric_type)(**metric_params)
