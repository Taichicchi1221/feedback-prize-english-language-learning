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
# main
# ====================================================
def main(cfg):
    seed_everything(cfg.globals.seed, deterministic=True)

    trace = Trace()

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
        checkpoint_callback = CheckPointCallback(
            filename=CHECKPOINT_NAME,
            monitor="valid_metric",
            mode=cfg.metric.mode,
            dirpath=".",
            auto_insert_metric_name=True,
            save_top_k=1,
            save_last=False,
            save_weights_only=True,
        )

        cfg.dataloader.steps_per_epoch = (len(train_dataloader) + cfg.trainer.accumulate_grad_batches - 1) // cfg.trainer.accumulate_grad_batches

        trainer = pl.Trainer(**cfg.trainer, callbacks=[checkpoint_callback])

        with trace.timer(f"trainer.fit fold{fold}"):
            trainer.fit(
                model,
                train_dataloaders=train_dataloader,
                val_dataloaders=valid_dataloader,
            )

        print(f"### best model path: {checkpoint_callback.best_model_path}")
        model_path_list.append(os.path.basename(checkpoint_callback.best_model_path))

        # plots
        os.makedirs("plots", exist_ok=True)
        plot_training_curve(
            train_history=model.history["train_metric"],
            valid_history=model.history["valid_metric"],
            filename=f"plots/training_curve_fold{fold}",
        )

        plot_lr_scheduler(
            lr_history=model.history["lr"],
            filename=f"plots/lr_scheduler_fold{fold}",
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
        filename="plots/oof_dist.png",
        target_names=TARGET_NAMES,
    )
    plot_scatter(
        preds=oof_df[TARGET_NAMES].to_numpy(),
        target=train_df[TARGET_NAMES].to_numpy(),
        filename="plots/oof_scatter.png",
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
