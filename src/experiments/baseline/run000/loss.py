from turtle import forward
import numpy as np
from sklearn.metrics import mean_squared_error

import torch
import torch.nn as nn
import torchmetrics

# ====================================================
# loss
# ====================================================
def get_loss(loss_type, loss_params):
    if loss_type == "MCRMSELoss":
        return eval(loss_type)(**loss_params)
    return getattr(torch.nn, loss_type)(**loss_params)


class MCRMSELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, preds, target):
        return torch.mean(torch.sqrt(torch.mean(torch.square(preds - target), dim=0)))


# ====================================================
# metric
# ====================================================
def MCRMSE(y_trues, y_preds):
    scores = []
    idxes = y_trues.shape[1]
    for i in range(idxes):
        y_true = y_trues[:, i]
        y_pred = y_preds[:, i]
        score = mean_squared_error(y_true, y_pred, squared=False)  # RMSE
        scores.append(score)
    mcrmse_score = np.mean(scores)
    return mcrmse_score, scores


class MCRMSEMetric(torchmetrics.Metric):
    full_state_update = False

    def __init__(self, num_cols):
        super().__init__()
        self.add_state("sum_values", default=torch.tensor([0] * num_cols, dtype=torch.float), dist_reduce_fx="sum")
        self.add_state("length", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, preds, target):
        assert preds.shape == target.shape
        self.sum_values += torch.sum((preds - target) ** 2, dim=0)
        self.length += len(target)

    def compute(self):
        return torch.mean(torch.sqrt(self.sum_values / self.length))


def get_metric(metric_type, metric_params):
    if metric_type == "MCRMSEMetric":
        return eval(metric_type)(**metric_params)
    return getattr(torch.nn, metric_type)(**metric_params)
