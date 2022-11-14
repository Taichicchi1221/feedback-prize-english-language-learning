import torch
import torch.nn.functional as F
from adabelief_pytorch import AdaBelief

import transformers

# ====================================================
# optimizer
# ====================================================
def get_optimizer(parameters, optimizer_type, optimizer_params):
    if optimizer_params is None:
        optimizer_params = {}

    if optimizer_type == "AdaBelief":
        return AdaBelief(parameters, **optimizer_params)

    return getattr(torch.optim, optimizer_type)(parameters, **optimizer_params)


# ====================================================
# scheduler
# ====================================================
def get_scheduler(optimizer, scheduler_type, scheduler_params):
    if scheduler_type is None:
        return None

    if scheduler_params is None:
        scheduler_params = {}

    # torch
    if hasattr(torch.optim.lr_scheduler, scheduler_type):
        return getattr(torch.optim.lr_scheduler, scheduler_type)(optimizer, **scheduler_params)

    # transformers
    if scheduler_type in [
        "linear",
        "cosine",
        "cosine_with_restarts",
        "polynomial",
        "constant",
        "constant_with_warmup",
    ]:
        return transformers.get_scheduler(
            name=scheduler_type,
            optimizer=optimizer,
            **scheduler_params,
        )


# ====================================================
# AWP
# ====================================================


class AWP:
    def __init__(
        self,
        adv_param="weight",
        adv_lr=1.0,
        adv_eps=0.01,
        start_epoch=0,
        adv_step=1,
    ):
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.start_epoch = start_epoch
        self.adv_step = adv_step
        self.backup = {}
        self.backup_eps = {}

    def attack_backward(self, batch, model, optimizer):
        self._save(model)
        self._attack_step(model)
        out = model(batch)
        adv_loss = model.criterion(out, batch["label"])
        optimizer.zero_grad()
        return adv_loss

    def _attack_step(self, model):
        e = 1e-6
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data = torch.min(torch.max(param.data, self.backup_eps[name][0]), self.backup_eps[name][1])
                # param.data.clamp_(*self.backup_eps[name])

    def _save(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.adv_eps * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

    def restore(self, model):
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}
