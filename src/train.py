import os
import gc
import re
import sys
import copy
import time
import glob
import math
import random
import psutil
import shutil
import typing
import numbers
import warnings
import argparse
from collections import defaultdict
from pathlib import Path
from datetime import datetime
from collections.abc import MutableMapping, Iterable
from abc import ABC, ABCMeta, abstractmethod

import json
import yaml

from box import Box
from omegaconf import OmegaConf

from tqdm.auto import tqdm

import joblib

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from sklearn.metrics import mean_squared_error

from scipy.special import expit as sigmoid

import transformers

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext
import torchmetrics

import pytorch_lightning as pl


os.environ["TOKENIZERS_PARALLELISM"] = "true"
tqdm.pandas()

##
from utils import *
from loss import get_loss, get_metric, MCRMSE
from optimizer import get_optimizer, get_scheduler
from tokenizer import get_tokenizer
from dataset import Dataset, Collate
from model import *


# ====================================================
# constants
# ====================================================
TARGET_NAMES = [
    "cohesion",
    "syntax",
    "vocabulary",
    "phraseology",
    "grammar",
    "conventions",
]

# ====================================================
# plots
# ====================================================
def plot_training_curve(train_history, valid_history, filename):
    plt.figure()
    legends = []
    plt.plot(range(len(train_history)), train_history, marker=".", color="skyblue")
    legends.append("train")
    plt.plot(range(len(valid_history)), valid_history, marker=".", color="orange")
    legends.append("valid")
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
    Fold = MultilabelStratifiedKFold(n_splits=n_fold, shuffle=shuffle, random_state=seed)
    for n, (train_index, valid_index) in enumerate(Fold.split(df, df[target_names])):
        df.loc[valid_index, "fold"] = int(n)
    df["fold"] = df["fold"].astype(int)

    print("# folds")
    print(df.groupby("fold").size())


# ====================================================
# main
# ====================================================
def main(cfg):
    pl.seed_everything(cfg.globals.seed)

    train_df = pd.read_csv("../input/feedback-prize-english-language-learning/train.csv")
    train_df.sort_values("text_id", inplace=True)

    if cfg.globals.debug:
        train_df = train_df.sample(300).reset_index(drop=True)
        cfg.globals.n_fold = 3
        cfg.trainer.max_epochs = 2

    prepare_fold(train_df, n_fold=cfg.globals.n_fold, target_names=TARGET_NAMES)
    oof_df_list = []
    model_path_list = []

    for fold in range(cfg.globals.n_fold):
        print("#" * 30, f"fold: {fold}", "#" * 30)
        # tokenizer
        tokenizer = get_tokenizer(cfg.tokenizer.path, cfg.tokenizer.params)

        # dataset
        train_dataset = Dataset(
            train_df.query(f"fold != {fold}"),
            tokenizer=tokenizer,
            max_length=cfg.tokenizer.max_length,
            target_names=TARGET_NAMES,
        )
        valid_dataset = Dataset(
            train_df.query(f"fold == {fold}"),
            tokenizer=tokenizer,
            max_length=cfg.tokenizer.max_length,
            target_names=TARGET_NAMES,
        )

        # dataloader
        collate_fn = Collate(tokenizer, max_length=cfg.tokenizer.max_length)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, collate_fn=collate_fn, **cfg.dataloader.train)
        valid_dataloader = torch.utils.data.DataLoader(valid_dataset, collate_fn=collate_fn, **cfg.dataloader.valid)

        model = Model(
            encoder_cfg=cfg.model.encoder,
            head_cfg=cfg.model.head,
            loss_cfg=cfg.loss,
            metric_cfg=cfg.metric,
            optimizer_cfg=cfg.optimizer,
            scheduler_cfg=cfg.scheduler,
        )

        transformers_config = copy.deepcopy(model.config)

        encoder_name = os.path.basename(cfg.model.encoder.path)
        CHECKPOINT_NAME = f"fold{fold}_{encoder_name}_" "{epoch:02d}_{step:03d}_{valid_metric:.3f}"
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            filename=CHECKPOINT_NAME,
            monitor="valid_metric",
            mode=cfg.metric.mode,
            dirpath="model",
            auto_insert_metric_name=True,
            save_top_k=1,
            save_last=False,
            save_weights_only=True,
        )

        cfg.dataloader.steps_per_epoch = (len(train_dataloader) + cfg.trainer.accumulate_grad_batches - 1) // cfg.trainer.accumulate_grad_batches

        trainer = pl.Trainer(**cfg.trainer, callbacks=[checkpoint_callback])
        trainer.fit(
            model,
            train_dataloaders=train_dataloader,
            val_dataloaders=valid_dataloader,
        )

        print(f"best model path: {checkpoint_callback.best_model_path}")
        model_path_list.append(checkpoint_callback.best_model_path)

        # plots
        plot_training_curve(
            train_history=model.history["train_metric"],
            valid_history=model.history["valid_metric"],
            filename=f"training_curve_fold{fold}",
        )

        plot_lr_scheduler(
            lr_history=model.history["lr"],
            filename=f"lr_scheduler_fold{fold}",
            steps_per_epoch=cfg.dataloader.steps_per_epoch,
            accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        )

        # oof
        model = EvalModel.load_from_checkpoint(
            checkpoint_callback.best_model_path,
            encoder_cfg=cfg.model.encoder,
            head_cfg=cfg.model.head,
        )
        oof_prediction_df = pd.DataFrame(
            torch.cat(trainer.predict(model=model, dataloaders=valid_dataloader)).numpy(),
            columns=TARGET_NAMES,
        )
        oof_prediction_df.insert(0, "text_id", valid_dataset.text_ids)
        oof_df_list.append(oof_prediction_df)

        oof_prediction_df.sort_values("text_id", inplace=True)
        val_df = train_df.query(f"fold == {fold}").sort_values("text_id")
        score, detail_score = MCRMSE(val_df[TARGET_NAMES].to_numpy(), oof_prediction_df[TARGET_NAMES].to_numpy())
        oof_score_result = {"oof_score": score}
        for i, target_name in enumerate(TARGET_NAMES):
            oof_score_result[f"oof_score_{target_name}"] = detail_score[i]
        print("#" * 30, f"oof score of fold{fold}", "#" * 30)
        print(oof_score_result)
        print("#" * 30, f"oof score of fold{fold}", "#" * 30)

        del train_dataset, valid_dataset, train_dataloader, valid_dataloader, trainer, model
        gc.collect()
        torch.cuda.empty_cache()
        pl.utilities.memory.garbage_collection_cuda()

    oof_df = pd.concat(oof_df_list)
    oof_df.sort_values("text_id", inplace=True)
    valid_score, valid_detail_score = MCRMSE(train_df[TARGET_NAMES].to_numpy(), oof_df[TARGET_NAMES].to_numpy())
    valid_score_result = {"valid_score": valid_score}
    for i, target_name in enumerate(TARGET_NAMES):
        valid_score_result[f"valid_score_{target_name}"] = valid_detail_score[i]
    valid_score_result["public_score"] = np.nan
    valid_score_result["private_score"] = np.nan
    print("#" * 30, f"valid score", "#" * 30)
    print(valid_score_result)
    print("#" * 30, f"valid score", "#" * 30)

    # plots
    plot_dist(
        preds=oof_df[TARGET_NAMES].to_numpy(),
        target=train_df[TARGET_NAMES].to_numpy(),
        filename="oof_dist.png",
        target_names=TARGET_NAMES,
    )
    plot_scatter(
        preds=oof_df[TARGET_NAMES].to_numpy(),
        target=train_df[TARGET_NAMES].to_numpy(),
        filename="oof_scatter.png",
        target_names=TARGET_NAMES,
    )

    # save results
    OmegaConf.save(cfg, "config.yaml")
    oof_df.to_csv("oof.csv", index=False)
    joblib.dump(model_path_list, "model_path_list.pkl")
    torch.save(transformers_config, "transformers_config.pth")
    tokenizer.save_pretrained("tokenizer")

    results = {
        "params": cfg,
        "metrics": valid_score_result,
    }
    joblib.dump(results, "results.pkl")
