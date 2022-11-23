import math

import torch
import torch.nn.functional as F

import transformers

# ====================================================
# optimizer
# ====================================================
# https://www.kaggle.com/code/rhtsingh/on-stability-of-few-sample-transformer-fine-tuning/notebook
class PriorWD(torch.optim.Optimizer):
    def __init__(self, optim, use_prior_wd=False, exclude_last_group=True):
        super(PriorWD, self).__init__(optim.param_groups, optim.defaults)
        self.param_groups = optim.param_groups
        self.optim = optim
        self.use_prior_wd = use_prior_wd
        self.exclude_last_group = exclude_last_group
        self.weight_decay_by_group = []
        for i, group in enumerate(self.param_groups):
            self.weight_decay_by_group.append(group["weight_decay"])
            group["weight_decay"] = 0

        self.prior_params = {}
        for i, group in enumerate(self.param_groups):
            for p in group["params"]:
                self.prior_params[id(p)] = p.detach().clone()

    def step(self, closure=None):
        if self.use_prior_wd:
            for i, group in enumerate(self.param_groups):
                for p in group["params"]:
                    if self.exclude_last_group and i == len(self.param_groups):
                        p.data.add_(-group["lr"] * self.weight_decay_by_group[i], p.data)
                    else:
                        p.data.add_(
                            -group["lr"] * self.weight_decay_by_group[i],
                            p.data - self.prior_params[id(p)],
                        )
        loss = self.optim.step(closure)

        return loss

    def compute_distance_to_prior(self, param):
        assert id(param) in self.prior_params, "parameter not in PriorWD optimizer"
        return (param.data - self.prior_params[id(param)]).pow(2).sum().sqrt()


# https://github.com/facebookresearch/madgrad/blob/main/madgrad/madgrad.py
class MADGRAD(torch.optim.Optimizer):
    """
    MADGRAD_: A Momentumized, Adaptive, Dual Averaged Gradient Method for Stochastic
    Optimization.
    .. _MADGRAD: https://arxiv.org/abs/2101.11075
    MADGRAD is a general purpose optimizer that can be used in place of SGD or
    Adam may converge faster and generalize better. Currently GPU-only.
    Typically, the same learning rate schedule that is used for SGD or Adam may
    be used. The overall learning rate is not comparable to either method and
    should be determined by a hyper-parameter sweep.
    MADGRAD requires less weight decay than other methods, often as little as
    zero. Momentum values used for SGD or Adam's beta1 should work here also.
    On sparse problems both weight_decay and momentum should be set to 0.
    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float):
            Learning rate (default: 1e-2).
        momentum (float):
            Momentum value in the range [0,1) (default: 0.9).
        weight_decay (float):
            Weight decay, i.e. a L2 penalty (default: 0).
        eps (float):
            Term added to the denominator outside of the root operation to improve
            numerical stability. (default: 1e-6).
            This parameter is less important in MADGRAD than in Adam.
            On problems with very small gradients, setting this to 0 will improve convergence.
        decouple_decay (bool):
            Apply AdamW style decoupled weight decay (EXPERIMENTAL).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-2,
        momentum: float = 0.9,
        weight_decay: float = 0,
        eps: float = 1e-6,
        decouple_decay=False,
    ):
        if momentum < 0 or momentum >= 1:
            raise ValueError(f"Momentum {momentum} must be in the range [0,1)")
        if lr <= 0:
            raise ValueError(f"Learning rate {lr} must be positive")
        if weight_decay < 0:
            raise ValueError(f"Weight decay {weight_decay} must be non-negative")
        if eps < 0:
            raise ValueError(f"Eps must be non-negative")

        defaults = dict(lr=lr, eps=eps, momentum=momentum, weight_decay=weight_decay, decouple_decay=decouple_decay)
        super().__init__(params, defaults)

    @property
    def supports_memory_efficient_fp16(self):
        return False

    @property
    def supports_flat_params(self):
        return True

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        # step counter must be stored in state to ensure correct behavior under
        # optimizer sharding
        if "k" not in self.state:
            self.state["k"] = torch.tensor([0], dtype=torch.long)
        k = self.state["k"].item()

        for group in self.param_groups:
            eps = group["eps"]
            lr = group["lr"] + eps
            decay = group["weight_decay"]
            momentum = group["momentum"]
            decouple_decay = group["decouple_decay"]

            ck = 1 - momentum
            lamb = lr * math.pow(k + 1, 0.5)

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                if "grad_sum_sq" not in state:
                    state["grad_sum_sq"] = torch.zeros_like(p.data).detach()
                    state["s"] = torch.zeros_like(p.data).detach()
                    if momentum != 0:
                        state["x0"] = torch.clone(p.data).detach()

                if momentum != 0.0 and grad.is_sparse:
                    raise RuntimeError("momentum != 0 is not compatible with sparse gradients")

                grad_sum_sq = state["grad_sum_sq"]
                s = state["s"]

                # Apply weight decay
                if decay != 0 and not decouple_decay:
                    if grad.is_sparse:
                        raise RuntimeError("weight_decay option is not compatible with sparse gradients")

                    grad.add_(p.data, alpha=decay)

                if grad.is_sparse:
                    grad = grad.coalesce()
                    grad_val = grad._values()

                    p_masked = p.sparse_mask(grad)
                    grad_sum_sq_masked = grad_sum_sq.sparse_mask(grad)
                    s_masked = s.sparse_mask(grad)

                    # Compute x_0 from other known quantities
                    rms_masked_vals = grad_sum_sq_masked._values().pow(1 / 3).add_(eps)
                    x0_masked_vals = p_masked._values().addcdiv(s_masked._values(), rms_masked_vals, value=1)

                    # Dense + sparse op
                    grad_sq = grad * grad
                    grad_sum_sq.add_(grad_sq, alpha=lamb)
                    grad_sum_sq_masked.add_(grad_sq, alpha=lamb)

                    rms_masked_vals = grad_sum_sq_masked._values().pow_(1 / 3).add_(eps)

                    if eps == 0:
                        rms_masked_vals[rms_masked_vals == 0] = float("inf")

                    s.add_(grad, alpha=lamb)
                    s_masked._values().add_(grad_val, alpha=lamb)

                    # update masked copy of p
                    p_kp1_masked_vals = x0_masked_vals.addcdiv(s_masked._values(), rms_masked_vals, value=-1)
                    # Copy updated masked p to dense p using an add operation
                    p_masked._values().add_(p_kp1_masked_vals, alpha=-1)
                    p.data.add_(p_masked, alpha=-1)
                else:
                    if momentum == 0:
                        # Compute x_0 from other known quantities
                        rms = grad_sum_sq.pow(1 / 3).add_(eps)
                        x0 = p.data.addcdiv(s, rms, value=1)
                    else:
                        x0 = state["x0"]

                    # Accumulate second moments
                    grad_sum_sq.addcmul_(grad, grad, value=lamb)
                    rms = grad_sum_sq.pow(1 / 3).add_(eps)

                    if eps == 0:
                        rms[rms == 0] = float("inf")

                    # Update s
                    s.data.add_(grad, alpha=lamb)

                    if decay != 0 and decouple_decay:
                        p_old = p.data.clone()

                    # Step
                    if momentum == 0:
                        p.data.copy_(x0.addcdiv(s, rms, value=-1))
                    else:
                        z = x0.addcdiv(s, rms, value=-1)

                        # p is a moving average of z
                        p.data.mul_(1 - ck).add_(z, alpha=ck)

                    if decay != 0 and decouple_decay:
                        p.data.add_(p_old, alpha=-lr * decay)

        self.state["k"] += 1
        return loss


def get_optimizer(parameters, optimizer_type, optimizer_params, pretrained_weight_decay):
    if optimizer_params is None:
        optimizer_params = {}

    if optimizer_type == "MADGRAD":
        optimizer = MADGRAD(parameters, **optimizer_params)
    elif hasattr(transformers.optimization, optimizer_type):
        optimizer = getattr(transformers.optimization, optimizer_type)(parameters, **optimizer_params)
    else:
        optimizer = getattr(torch.optim, optimizer_type)(parameters, **optimizer_params)

    if pretrained_weight_decay:
        return PriorWD(optimizer)

    return optimizer


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
