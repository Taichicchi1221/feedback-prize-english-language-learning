import torch

# ====================================================
# optimizer
# ====================================================
def get_optimizer(parameters, optimizer_type, optimizer_params):
    if optimizer_params is None:
        optimizer_params = {}

    return getattr(torch.optim, optimizer_type)(parameters, **optimizer_params)


# ====================================================
# scheduler
# ====================================================
def get_scheduler(optimizer, scheduler_type, scheduler_params):
    if scheduler_params is None:
        scheduler_params = {}

    return getattr(torch.optim.lr_scheduler, scheduler_type)(optimizer, **scheduler_params)
