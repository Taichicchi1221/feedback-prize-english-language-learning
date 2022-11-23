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


os.environ["TOKENIZERS_PARALLELISM"] = "false"
tqdm.pandas()

##
from utils import *
from loss import get_loss, get_metric, MCRMSE
from optimizer import get_optimizer, get_scheduler
from tokenizer import get_tokenizer
from dataset import Dataset, Collate
from model import *
from preprocess import Preprocessor

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


def train_function(
    train_df,
    valid_df,
    tokenizer,
    preprocess_method_name,
    train_max_length,
    valid_max_length,
    train_collate_function,
    valid_collate_function,
    train_dataloader_args,
    valid_dataloader_args,
    model,
    trainer_args,
    checkpoint_name=None,
    swa_cfg=None,
):
    trace = Trace()

    model.train()

    # dataset
    train_preprocessor = Preprocessor(tokenizer=tokenizer, max_length=train_max_length)
    train_dataset = Dataset(
        train_df,
        tokenizer=tokenizer,
        max_length=train_max_length,
        target_names=TARGET_NAMES,
        preprocess_func=getattr(train_preprocessor, preprocess_method_name),
    )
    train_collate_function = Collate(tokenizer, max_length=train_max_length)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        collate_fn=train_collate_function,
        **train_dataloader_args,
    )

    if valid_df is not None:
        valid_preprocessor = Preprocessor(tokenizer=tokenizer, max_length=valid_max_length)
        valid_dataset = Dataset(
            valid_df,
            tokenizer=tokenizer,
            max_length=valid_max_length,
            target_names=TARGET_NAMES,
            preprocess_func=getattr(valid_preprocessor, preprocess_method_name),
        )
        valid_collate_function = Collate(tokenizer, max_length=valid_max_length)
        valid_dataloader = torch.utils.data.DataLoader(
            valid_dataset,
            collate_fn=valid_collate_function,
            **valid_dataloader_args,
        )
    else:
        valid_dataloader = None

    with trace.timer(f"trainer.fit"):
        callbacks = []
        if checkpoint_name:
            checkpoint_callback = CheckPointCallback(
                filename=checkpoint_name,
                monitor="valid_metric",
                mode="min",
                dirpath=".",
                auto_insert_metric_name=True,
                save_top_k=1,
                save_last=False,
                save_weights_only=True,
            )
            callbacks.append(checkpoint_callback)

        if swa_cfg is not None:
            swa_callback = pl.callbacks.StochasticWeightAveraging(**swa_cfg)
            callbacks.append(swa_callback)

        trainer = pl.Trainer(
            **trainer_args,
            callbacks=callbacks,
            logger=None,
        )
        trainer.fit(
            model,
            train_dataloaders=train_dataloader,
            val_dataloaders=valid_dataloader,
        )

    if checkpoint_name:
        best_model_path = checkpoint_callback.best_model_path
        print(f"### best model path: {best_model_path}")
        return best_model_path


def predict_function(
    df,
    tokenizer,
    preprocess_method_name,
    max_length,
    collate_function,
    dataloader_args,
    model,
    trainer_args,
):
    preprocessor = Preprocessor(tokenizer=tokenizer, max_length=max_length)
    dataset = Dataset(
        df=df,
        tokenizer=tokenizer,
        max_length=max_length,
        target_names=TARGET_NAMES,
        preprocess_func=getattr(preprocessor, preprocess_method_name),
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collate_function,
        **dataloader_args,
    )
    predict_trainer = pl.Trainer(**trainer_args, logger=None)
    model.eval()
    return torch.cat(predict_trainer.predict(model=model, dataloaders=dataloader)).numpy()


# ====================================================
# main
# ====================================================
def main(cfg):
    seed_everything(cfg.globals.seed, deterministic=True)

    train_df = pd.read_csv("../input/feedback-prize-english-language-learning/train.csv")
    train_df.sort_values("text_id", inplace=True)

    if cfg.globals.debug:
        train_df = train_df.sample(300).reset_index(drop=True)
        cfg.globals.n_fold = 3
        cfg.globals.epochs = 2

    # save config
    transformers_config = transformers.AutoConfig.from_pretrained(cfg.model.encoder.path)
    transformers_config.update(cfg.model.encoder.params)

    prepare_fold(train_df, n_fold=cfg.globals.n_fold, target_names=TARGET_NAMES)
    oof_df_list = []
    val_df_list = []
    model_path_list = []

    for fold in range(cfg.globals.n_fold):
        if cfg.globals.use_folds is not None and fold not in cfg.globals.use_folds:
            continue

        print("#" * 30, f"fold: {fold}", "#" * 30)
        train = train_df.query(f"fold != {fold}").reset_index(drop=True)
        valid = train_df.query(f"fold == {fold}").reset_index(drop=True)

        CHECKPOINT_NAME = f"fold{fold}_{os.path.basename(cfg.model.encoder.path)}_" "{epoch:02d}_{step:03d}_{valid_metric:.3f}"

        tokenizer = get_tokenizer(tokenizer_path=cfg.tokenizer.path, tokenizer_params=cfg.tokenizer.params)
        train_collate_function = Collate(tokenizer=tokenizer, max_length=cfg.tokenizer.max_length.train)
        valid_collate_function = Collate(tokenizer=tokenizer, max_length=cfg.tokenizer.max_length.test)

        model = Model(
            config_cfg=cfg.config,
            encoder_cfg=cfg.model.encoder,
            head_cfg=cfg.model.head,
            loss_cfg=cfg.loss,
            metric_cfg=cfg.metric,
            optimizer_cfg=cfg.optimizer,
            scheduler_cfg=cfg.scheduler,
            awp_cfg=cfg.awp,
            pretrained=True,
        )

        cfg.globals.steps_per_epoch = calc_steps_per_epoch(
            len_dataset=len(train),
            batch_size=cfg.dataloader.train.batch_size,
            accumulate_grad_batches=cfg.optimizer.accumulate_grad_batches,
        )
        cfg.globals.total_steps = cfg.globals.steps_per_epoch * cfg.globals.epochs

        best_model_path = train_function(
            train_df=train,
            valid_df=valid,
            tokenizer=tokenizer,
            preprocess_method_name=cfg.preprocessor.method,
            train_max_length=cfg.tokenizer.max_length.train,
            valid_max_length=cfg.tokenizer.max_length.test,
            train_collate_function=train_collate_function,
            valid_collate_function=valid_collate_function,
            train_dataloader_args=cfg.dataloader.train,
            valid_dataloader_args=cfg.dataloader.test,
            model=model,
            trainer_args=cfg.trainer.train,
            swa_cfg=cfg.swa,
            checkpoint_name=CHECKPOINT_NAME,
        )
        model_path_list.append(os.path.basename(best_model_path))

        # plots
        os.makedirs("plots", exist_ok=True)
        plot_training_curve(
            train_history=model.history["train_metric"],
            valid_history=model.history["valid_metric"],
            filename=f"plots/training_curve_fold{fold}.png",
        )

        plot_lr_scheduler(
            lr_history=model.history["lr"],
            filename=f"plots/lr_scheduler_fold{fold}.png",
            steps_per_epoch=cfg.globals.steps_per_epoch,
            accumulate_grad_batches=cfg.optimizer.accumulate_grad_batches,
        )

        # oof
        model = EvalModel.load_from_checkpoint(
            best_model_path,
            config_cfg=cfg.config,
            encoder_cfg=cfg.model.encoder,
            head_cfg=cfg.model.head,
            loss_cfg=cfg.loss,
            pretrained=False,
        )

        oof_prediction = predict_function(
            df=valid,
            tokenizer=tokenizer,
            preprocess_method_name=cfg.preprocessor.method,
            max_length=cfg.tokenizer.max_length.test,
            collate_function=valid_collate_function,
            dataloader_args=cfg.dataloader.test,
            model=model,
            trainer_args=cfg.trainer.predict,
        )

        oof_prediction_df = pd.concat(
            [
                pd.DataFrame({"text_id": valid["text_id"].to_numpy(), "full_text": valid["full_text"].to_numpy()}),
                pd.DataFrame({TARGET_NAMES[i]: oof_prediction[:, i] for i in range(len(TARGET_NAMES))}),
            ],
            axis=1,
        )

        ### pseudo labeling
        if cfg.globals.pseudo_label_epochs:
            pseudo_labeling_train_df = pd.concat([train, oof_prediction_df], axis=0).reset_index(drop=True)

            cfg.globals.steps_per_epoch = calc_steps_per_epoch(
                len_dataset=len(pseudo_labeling_train_df),
                batch_size=cfg.dataloader.train.batch_size,
                accumulate_grad_batches=cfg.pseudo_label_optimizer.accumulate_grad_batches,
            )
            cfg.globals.total_steps = cfg.globals.steps_per_epoch * cfg.globals.pseudo_label_epochs

            model = Model.load_from_checkpoint(
                best_model_path,
                config_cfg=cfg.config,
                encoder_cfg=cfg.model.encoder,
                head_cfg=cfg.model.head,
                loss_cfg=cfg.loss,
                metric_cfg=cfg.metric,
                optimizer_cfg=cfg.pseudo_label_optimizer,
                scheduler_cfg=cfg.pseudo_label_scheduler,
                pretrained=False,
            )

            train_function(
                train_df=pseudo_labeling_train_df,
                valid_df=None,
                tokenizer=tokenizer,
                preprocess_method_name=cfg.preprocessor.method,
                train_max_length=cfg.tokenizer.max_length.train,
                valid_max_length=None,
                train_collate_function=train_collate_function,
                valid_collate_function=None,
                train_dataloader_args=cfg.dataloader.train,
                valid_dataloader_args=None,
                model=model,
                trainer_args=cfg.trainer.pseudo_label_train,
                checkpoint_name=None,
            )

            # oof
            oof_prediction = predict_function(
                df=valid,
                tokenizer=tokenizer,
                preprocess_method_name=cfg.preprocessor.method,
                max_length=cfg.tokenizer.max_length.test,
                collate_function=valid_collate_function,
                dataloader_args=cfg.dataloader.test,
                model=model,
                trainer_args=cfg.trainer.predict,
            )
            oof_prediction_df = pd.concat(
                [
                    pd.DataFrame({"text_id": valid["text_id"].to_numpy(), "full_text": valid["full_text"].to_numpy()}),
                    pd.DataFrame({TARGET_NAMES[i]: oof_prediction[:, i] for i in range(len(TARGET_NAMES))}),
                ],
                axis=1,
            )

        oof_df_list.append(oof_prediction_df)
        val_df = train_df.query(f"fold == {fold}").sort_values("text_id").reset_index(drop=True)
        val_df_list.append(val_df)

        oof_prediction_df.sort_values("text_id", inplace=True)
        score, detail_score = MCRMSE(val_df[TARGET_NAMES].to_numpy(), oof_prediction_df[TARGET_NAMES].to_numpy())
        oof_score_result = {"oof_score": score}
        for i, target_name in enumerate(TARGET_NAMES):
            oof_score_result[f"oof_score_{target_name}"] = detail_score[i]
        print("#" * 30, f"oof score of fold{fold}", "#" * 30)
        print(oof_score_result)
        print("#" * 30, f"oof score of fold{fold}", "#" * 30)

        gc.collect()
        torch.cuda.empty_cache()
        pl.utilities.memory.garbage_collection_cuda()

    val_df = pd.concat(val_df_list, axis=0)
    val_df.sort_values("text_id", inplace=True)

    oof_df = pd.concat(oof_df_list, axis=0)
    oof_df.sort_values("text_id", inplace=True)

    valid_score, valid_detail_score = MCRMSE(val_df[TARGET_NAMES].to_numpy(), oof_df[TARGET_NAMES].to_numpy())
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
        target=val_df[TARGET_NAMES].to_numpy(),
        filename="plots/oof_dist.png",
        target_names=TARGET_NAMES,
    )
    plot_scatter(
        preds=oof_df[TARGET_NAMES].to_numpy(),
        target=val_df[TARGET_NAMES].to_numpy(),
        filename="plots/oof_scatter.png",
        target_names=TARGET_NAMES,
    )

    # save results
    OmegaConf.save(cfg, "config.yaml")
    oof_df[["text_id"] + TARGET_NAMES].to_csv("oof.csv", index=False)
    joblib.dump(model_path_list, "model_path_list.pkl")
    transformers_config.save_pretrained("config_tokenizer")
    tokenizer.save_pretrained("config_tokenizer")

    results = {
        "params": OmegaConf.to_container(cfg),
        "metrics": valid_score_result,
    }
    joblib.dump(results, "results.pkl")
