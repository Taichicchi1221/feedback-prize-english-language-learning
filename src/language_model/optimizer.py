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
        adv_lr=1,
        adv_eps=0.2,
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

    def attack_backward(self, model, batch, epoch):
        if (self.adv_lr == 0) or (epoch < self.start_epoch):
            return None

        optimizer = model.optimizers()

        self._save(model)
        for i in range(self.adv_step):
            y = batch["label"]
            self._attack_step(model)
            y_hat = model(batch)
            loss = model.criterion(y_hat, y)
            optimizer.zero_grad()
            model.manual_backward(loss)

        self._restore(model)

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

    def _restore(self, model):
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}


# ====================================================
# SiFT
# ====================================================
class PerturbationLayer(torch.nn.Module):
    def __init__(self, hidden_size, learning_rate=1e-4, init_perturbation=1e-2):
        super().__init__()
        self.learning_rate = learning_rate
        self.init_perturbation = init_perturbation
        self.delta = None
        self.LayerNorm = torch.nn.LayerNorm(hidden_size, 1e-7, elementwise_affine=False)
        self.adversarial_mode = False

    def adversarial_(self, adversarial=True):
        self.adversarial_mode = adversarial
        if not adversarial:
            self.delta = None

    def forward(self, input):
        if not self.adversarial_mode:
            self.input = self.LayerNorm(input)
            return self.input
        else:
            if self.delta is None:
                self.update_delta(requires_grad=True)
            return self.perturbated_input

    def update_delta(self, requires_grad=False):
        if not self.adversarial_mode:
            return True
        if self.delta is None:
            delta = torch.clamp(
                self.input.new(self.input.size()).normal_(0, self.init_perturbation).float(),
                -2 * self.init_perturbation,
                2 * self.init_perturbation,
            )
        else:
            grad = self.delta.grad
            self.delta.grad = None
            delta = self.delta
            norm = grad.norm()
            if torch.isnan(norm) or torch.isinf(norm):
                return False
            eps = self.learning_rate
            with torch.no_grad():
                delta = delta + eps * grad / (1e-6 + grad.abs().max(-1, keepdim=True)[0])
        self.delta = delta.float().detach().requires_grad_(requires_grad)
        self.perturbated_input = (self.input.to(delta).detach() + self.delta).to(self.input)
        return True


def hook_sift_layer(
    model,
    hidden_size,
    learning_rate=1e-4,
    init_perturbation=1e-2,
    target_module="embeddings.LayerNorm",
):
    """
    Hook the sift perturbation layer to and existing model. With this method, you can apply adversarial training
    without changing the existing model implementation.
    Params:
      `model`: The model instance to apply adversarial training
      `hidden_size`: The dimmension size of the perturbated embedding
      `learning_rate`: The learning rate to update the perturbation
      `init_perturbation`: The initial range of perturbation
      `target_module`: The module to apply perturbation. It can be the name of the sub-module of the model or the sub-module instance.
      The perturbation layer will be inserted before the sub-module.
    Outputs:
      The perturbation layers.
    """

    if isinstance(target_module, str):
        _modules = [k for n, k in model.named_modules() if target_module in n]
    else:
        assert isinstance(target_module, torch.nn.Module), f"{type(target_module)} is not an instance of torch.nn.Module"
        _modules = [target_module]
    adv_modules = []
    for m in _modules:
        adv = PerturbationLayer(hidden_size, learning_rate, init_perturbation)

        def adv_hook(module, inputs):
            return adv(inputs[0])

        for h in list(m._forward_pre_hooks.keys()):
            if m._forward_pre_hooks[h].__name__ == "adv_hook":
                del m._forward_pre_hooks[h]
        m.register_forward_pre_hook(adv_hook)
        adv_modules.append(adv)
    return adv_modules


class AdversarialLearner:
    """Adversarial Learner
    This class is the helper class for adversarial training.
    Params:
      `model`: The model instance to apply adversarial training
      `perturbation_modules`: The sub modules in the model that will generate perturbations. If it's `None`,
      the constructor will detect sub-modules of type `PerturbationLayer` in the model.
    Example usage:
    ```python
    # Create DeBERTa model
    adv_modules = hook_sift_layer(model, hidden_size=768)
    adv = AdversarialLearner(model, adv_modules)
    def logits_fn(model, *wargs, **kwargs):
      logits,_ = model(*wargs, **kwargs)
      return logits
    logits,loss = model(**data)
    loss = loss + adv.loss(logits, logits_fn, **data)
    # Other steps is the same as general training.
    ```
    """

    def __init__(self, model, adv_modules=None):
        if adv_modules is None:
            self.adv_modules = [m for m in model.modules() if isinstance(m, PerturbationLayer)]
        else:
            self.adv_modules = adv_modules
        self.parameters = [p for p in model.parameters()]
        self.model = model
        self.perturbation_loss_fns = {
            "symmetric-kl": symmetric_kl,
            "kl": kl,
            "mse": mse,
        }

    def loss(self, target, logits_fn, loss_fn="symmetric-kl", *wargs, **kwargs):
        """
        Calculate the adversarial loss based on the given logits fucntion and loss function.
        Inputs:
        `target`: the logits from original inputs.
        `logits_fn`: the function that produces logits based on perturbated inputs. E.g.,
        ```python
        def logits_fn(model, *wargs, **kwargs):
          logits = model(*wargs, **kwargs)
          return logits
        ```
        `loss_fn`: the function that caclulate the loss from perturbated logits and target logits.
          - If it's a string, it can be pre-built loss functions, i.e. kl, symmetric_kl, mse.
          - If it's a function, it will be called to calculate the loss, the signature of the function will be,
          ```python
          def loss_fn(source_logits, target_logits):
            # Calculate the loss
            return loss
          ```
        `*wargs`: the positional arguments that will be passed to the model
        `**kwargs`: the key-word arguments that will be passed to the model
        Outputs:
          The loss based on pertubated inputs.
        """
        self.prepare()
        if isinstance(loss_fn, str):
            loss_fn = self.perturbation_loss_fns[loss_fn]
        pert_logits = logits_fn(self.model, *wargs, **kwargs)
        pert_loss = loss_fn(pert_logits, target.detach()).sum()
        pert_loss.backward()
        for m in self.adv_modules:
            ok = m.update_delta(True)

        for r, p in zip(self.prev, self.parameters):
            p.requires_grad_(r)
        pert_logits = logits_fn(self.model, *wargs, **kwargs)
        pert_loss = symmetric_kl(pert_logits, target)

        self.cleanup()
        return pert_loss.mean()

    def prepare(self):
        self.prev = [p.requires_grad for p in self.parameters]
        for p in self.parameters:
            p.requires_grad_(False)
        for m in self.adv_modules:
            m.adversarial_(True)

    def cleanup(self):
        for r, p in zip(self.prev, self.parameters):
            p.requires_grad_(r)

        for m in self.adv_modules:
            m.adversarial_(False)


def symmetric_kl(logits, target):
    logit_stu = logits.view(-1, logits.size(-1)).float()
    logit_tea = target.view(-1, target.size(-1)).float()
    logprob_stu = F.log_softmax(logit_stu, -1)
    logprob_tea = F.log_softmax(logit_tea, -1)
    prob_tea = logprob_tea.exp().detach()
    prob_stu = logprob_stu.exp().detach()
    floss = (prob_tea * (-logprob_stu)).sum(-1)  # Cross Entropy
    bloss = (prob_stu * (-logprob_tea)).sum(-1)  # Cross Entropy
    loss = floss + bloss
    return loss


def kl(logits, target):
    logit_stu = logits.view(-1, logits.size(-1)).float()
    logit_tea = target.view(-1, target.size(-1)).float()
    logprob_stu = F.log_softmax(logit_stu, -1)
    logprob_tea = F.log_softmax(logit_tea.detach(), -1)
    prob_tea = logprob_tea.exp()
    loss = (prob_tea * (-logprob_stu)).sum(-1)  # Cross Entropy
    return loss


def mse(logits, target):
    logit_stu = logits.view(-1, logits.size(-1)).float()
    logit_tea = target.view(-1, target.size(-1)).float()
    return F.mse_loss(logit_stu.view(-1), logit_tea.view(-1))
