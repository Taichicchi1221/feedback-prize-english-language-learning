import os
import sys
import re
import time
import math
import random
import typing
import argparse
import psutil
import subprocess
from contextlib import contextmanager

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

import torch
import pytorch_lightning as pl

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


# ====================================================
# utils
# ====================================================


def flatten_dict(params: typing.Dict[typing.Any, typing.Any], delimiter: str = "/") -> typing.Dict[str, typing.Any]:
    """
    https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pytorch_lightning/loggers/base.py
    Flatten hierarchical dict, e.g. ``{'a': {'b': 'c'}} -> {'a/b': 'c'}``.
    Args:
        params: Dictionary containing the hyperparameters
        delimiter: Delimiter to express the hierarchy. Defaults to ``'/'``.
    Returns:
        Flattened dict.
    Examples:
        >>> LightningLoggerBase._flatten_dict({'a': {'b': 'c'}})
        {'a/b': 'c'}
        >>> LightningLoggerBase._flatten_dict({'a': {'b': 123}})
        {'a/b': 123}
        >>> LightningLoggerBase._flatten_dict({5: {'a': 123}})
        {'5/a': 123}
    """

    def _dict_generator(input_dict, prefixes=None):
        prefixes = prefixes[:] if prefixes else []
        if isinstance(input_dict, typing.MutableMapping):
            for key, value in input_dict.items():
                key = str(key)
                if isinstance(value, (typing.MutableMapping, argparse.Namespace)):
                    value = vars(value) if isinstance(value, argparse.Namespace) else value
                    yield from _dict_generator(value, prefixes + [key])
                else:
                    yield prefixes + [key, value if value is not None else str(None)]
        else:
            yield prefixes + [input_dict if input_dict is None else str(input_dict)]

    return {delimiter.join(keys): val for *keys, val in _dict_generator(params)}


def memory_used_to_str():
    pid = os.getpid()
    processs = psutil.Process(pid)
    memory_use = processs.memory_info()[0] / 2.0**30
    return "ram memory gb :" + str(np.round(memory_use, 2))


def seed_everything(seed: int = 42, deterministic: bool = False):
    """Set seeds"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = deterministic


def get_gpu_memory(cmd_path="nvidia-smi", target_properties=("memory.total", "memory.used")):
    """
    ref: https://www.12-technology.com/2022/01/pythongpu.html
    Returns
    -------
    gpu_total : ndarray,  "memory.total"
    gpu_used: ndarray, "memory.used"
    """

    # formatオプション定義
    format_option = "--format=csv,noheader,nounits"

    # コマンド生成
    cmd = "%s --query-gpu=%s %s" % (cmd_path, ",".join(target_properties), format_option)

    # サブプロセスでコマンド実行
    cmd_res = subprocess.check_output(cmd, shell=True)

    # コマンド実行結果をオブジェクトに変換
    gpu_lines = cmd_res.decode().split("\n")[0].split(", ")

    gpu_total = int(gpu_lines[0]) / 1024
    gpu_used = int(gpu_lines[1]) / 1024

    gpu_total = np.round(gpu_used, 1)
    gpu_used = np.round(gpu_used, 1)
    return gpu_total, gpu_used


class Trace:
    cuda = torch.cuda.is_available()
    if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
        is_competition_rerun = True
    else:
        is_competition_rerun = False

    @contextmanager
    def timer(self, title):
        t0 = time.time()
        p = psutil.Process(os.getpid())
        cpu_m0 = p.memory_info().rss / 2.0**30
        if self.cuda:
            gpu_m0 = get_gpu_memory()[0]
        yield
        cpu_m1 = p.memory_info().rss / 2.0**30
        if self.cuda:
            gpu_m1 = get_gpu_memory()[0]

        cpu_delta = cpu_m1 - cpu_m0
        if self.cuda:
            gpu_delta = gpu_m1 - gpu_m0

        cpu_sign = "+" if cpu_delta >= 0 else "-"
        cpu_delta = math.fabs(cpu_delta)

        if self.cuda:
            gpu_sign = "+" if gpu_delta >= 0 else "-"
        if self.cuda:
            gpu_delta = math.fabs(gpu_delta)

        cpu_message = f"{cpu_m1:.1f}GB({cpu_sign}{cpu_delta:.1f}GB)"
        if self.cuda:
            gpu_message = f"{gpu_m1:.1f}GB({gpu_sign}{gpu_delta:.1f}GB)"

        if self.cuda:
            message = f"[cpu: {cpu_message}, gpu: {gpu_message}: {time.time() - t0:.1f}sec] {title} "
        else:
            message = f"[cpu: {cpu_message}: {time.time() - t0:.1f}sec] {title} "

        print(message, file=sys.stderr)


def calc_steps_per_epoch(len_dataset, batch_size, accumulate_grad_batches):
    steps_per_epoch = (len_dataset // batch_size + accumulate_grad_batches - 1) // accumulate_grad_batches + 1
    return steps_per_epoch


# ====================================================
# plots
# ====================================================
def plot_training_curve(train_history, valid_history, filename, ymax=0.7, ymin=0.0):
    plt.figure()
    legends = []
    plt.plot(range(len(train_history)), train_history, marker=".", color="skyblue")
    legends.append("train")
    plt.plot(range(len(valid_history)), valid_history, marker=".", color="orange")
    legends.append("valid")
    plt.ylim(bottom=ymin, top=ymax)
    plt.legend(legends)
    plt.savefig(filename)
    plt.close()


def plot_lr_scheduler(lr_history, filename, steps_per_epoch, accumulate_grad_batches):
    epoch_index = [step for step in range(len(lr_history)) if step % (steps_per_epoch * accumulate_grad_batches) == 0]
    plt.figure()
    plt.plot(range(len(lr_history)), lr_history)
    plt.plot(
        [i for i in range(len(lr_history)) if i in epoch_index],
        [lr_history[i] for i in range(len(lr_history)) if i in epoch_index],
        color="orange",
        linestyle="None",
        marker="D",
    )
    plt.xlabel("step")
    plt.ylabel("lr")
    plt.legend(["lr", "epoch"])
    plt.savefig(filename)
    plt.close()


def plot_dist(preds, target, filename, target_names):
    assert preds.shape == target.shape
    nrows = int(np.sqrt(target.shape[1]))
    ncols = (target.shape[1] + nrows - 1) // nrows
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 9))
    fig.tight_layout()
    for i, target_name in enumerate(target_names):
        r, c = i // ncols, i % ncols
        ax[r, c].set_title(target_name)
        ax[r, c].hist(target[:, i], bins=100, color="orange", alpha=0.5, log=True, label="target")
        ax[r, c].hist(preds[:, i], bins=100, color="skyblue", alpha=0.5, log=True, label="prediction")
        ax[r, c].legend()
    plt.subplots_adjust(hspace=0.2, wspace=0.2, left=0.05, right=0.95, bottom=0.05, top=0.95)
    fig.savefig(filename)
    plt.close()


def plot_scatter(preds, target, filename, target_names):
    assert preds.shape == target.shape
    nrows = int(np.sqrt(target.shape[1]))
    ncols = (target.shape[1] + nrows - 1) // nrows
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 9))
    fig.tight_layout()
    for i, target_name in enumerate(target_names):
        r, c = i // ncols, i % ncols
        ax[r, c].set_title(target_name)
        ax[r, c].scatter(target[:, i], preds[:, i])
        ax[r, c].set_xlabel("target")
        ax[r, c].set_ylabel("prediction")
    plt.subplots_adjust(hspace=0.2, wspace=0.2, left=0.05, right=0.95, bottom=0.05, top=0.95)
    fig.savefig(filename)
    plt.close()


# ====================================================
# data processing
# ====================================================
def prepare_fold(df, n_fold, target_names=None, seed=None):
    assert target_names is not None, "target_names must be set."
    shuffle = seed is not None
    kfold = MultilabelStratifiedKFold(n_splits=n_fold, shuffle=shuffle, random_state=seed)
    folds = np.zeros((len(df),), dtype=np.int32)
    y = pd.get_dummies(df[target_names], columns=target_names)
    for n, (train_index, valid_index) in enumerate(kfold.split(df, y)):
        folds[valid_index] = int(n)
    df["fold"] = folds

    print("# folds")
    print(df.groupby("fold").size())


# ====================================================
# checkpoint
# ====================================================
class CheckPointCallback(pl.callbacks.ModelCheckpoint):
    METRIC_SEP_CHAR = "-"

    @classmethod
    def _format_checkpoint_name(
        cls,
        filename: typing.Optional[str],
        metrics: typing.Dict[str, torch.Tensor],
        prefix: str = "",
        auto_insert_metric_name: bool = True,
    ) -> str:
        if not filename:
            # filename is not set, use default name
            filename = "{epoch}" + cls.CHECKPOINT_JOIN_CHAR + "{step}"

        # check and parse user passed keys in the string
        groups = re.findall(r"(\{.*?)[:\}]", filename)
        if len(groups) >= 0:
            for group in groups:
                name = group[1:]

                if auto_insert_metric_name:
                    filename = filename.replace(group, name + cls.METRIC_SEP_CHAR + "{" + name)

                # support for dots: https://stackoverflow.com/a/7934969
                filename = filename.replace(group, f"{{0[{name}]")

                if name not in metrics:
                    metrics[name] = torch.tensor(0)
            filename = filename.format(metrics)

        if prefix:
            filename = cls.CHECKPOINT_JOIN_CHAR.join([prefix, filename])

        return filename
