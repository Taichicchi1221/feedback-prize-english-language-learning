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
from loss import get_loss, get_metric
from optimizer import get_optimizer, get_scheduler
from tokenizer import get_tokenizer
from dataset import Dataset, Collate
from model import *
from train import train_function, predict_function

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
# cli args
# ====================================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=".")
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--work_dir", default=".")

    return parser.parse_args()


# ====================================================
# main
# ====================================================
def main():
    args = parse_args()

    cfg = OmegaConf.load(os.path.join(args.input_dir, "config.yaml"))
    model_path_list = joblib.load(os.path.join(args.input_dir, "model_path_list.pkl"))
    cfg.config.path = os.path.join(args.input_dir, "config_tokenizer")
    cfg.tokenizer.path = os.path.join(args.input_dir, "config_tokenizer")

    seed_everything(cfg.globals.seed, deterministic=True)

    train_df = pd.read_csv(os.path.join(args.work_dir, "../input/feedback-prize-english-language-learning/train.csv"))
    prepare_fold(train_df, n_fold=cfg.globals.n_fold, target_names=TARGET_NAMES)

    test_df = pd.read_csv(os.path.join(args.work_dir, "../input/feedback-prize-english-language-learning/test.csv"))
    for target_name in TARGET_NAMES:
        test_df[target_name] = -1

    submission_df_list = []
    for fold, model_path in enumerate(model_path_list):
        print("#" * 30, f"model_path: {model_path}", "#" * 30)

        train = train_df.copy()
        test = test_df.copy()

        tokenizer = get_tokenizer(tokenizer_path=cfg.tokenizer.path, tokenizer_params=cfg.tokenizer.params)
        train_collate_function = Collate(tokenizer=tokenizer, max_length=cfg.tokenizer.max_length.train)
        valid_collate_function = Collate(tokenizer=tokenizer, max_length=cfg.tokenizer.max_length.test)

        model = EvalModel.load_from_checkpoint(
            os.path.join(args.input_dir, model_path),
            config_cfg=cfg.config,
            encoder_cfg=cfg.model.encoder,
            head_cfg=cfg.model.head,
            loss_cfg=cfg.loss,
            pretrained=False,
        )

        test_prediction = predict_function(
            df=test,
            tokenizer=tokenizer,
            preprocess_method_name=cfg.preprocessor.method,
            max_length=cfg.tokenizer.max_length.test,
            collate_function=valid_collate_function,
            dataloader_args=cfg.dataloader.test,
            model=model,
            trainer_args=cfg.trainer.predict,
        )

        submission_prediction_df = pd.concat(
            [
                pd.DataFrame({"text_id": test["text_id"], "full_text": test["full_text"]}),
                pd.DataFrame({TARGET_NAMES[i]: test_prediction[:, i] for i in range(len(TARGET_NAMES))}),
            ],
            axis=1,
        )

        ### pseudo labeling
        if cfg.globals.pseudo_label_epochs:
            pseudo_labeling_train_df = pd.concat([train, submission_prediction_df], axis=0).reset_index(drop=True)
            print("#" * 100)
            print(pseudo_labeling_train_df)
            print("#" * 100)

            cfg.globals.steps_per_epoch = calc_steps_per_epoch(
                len_dataset=len(pseudo_labeling_train_df),
                batch_size=cfg.dataloader.train.batch_size,
                accumulate_grad_batches=cfg.trainer.pseudo_label_train.accumulate_grad_batches,
            )
            cfg.globals.total_steps = cfg.globals.steps_per_epoch * cfg.globals.pseudo_label_epochs

            model = Model.load_from_checkpoint(
                model_path,
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

            # prediction
            test_prediction = predict_function(
                df=test,
                tokenizer=tokenizer,
                preprocess_method_name=cfg.preprocessor.method,
                max_length=cfg.tokenizer.max_length.test,
                collate_function=valid_collate_function,
                dataloader_args=cfg.dataloader.test,
                model=model,
                trainer_args=cfg.trainer.predict,
            )

            submission_prediction_df = pd.concat(
                [
                    pd.DataFrame({"text_id": test["text_id"], "text": test["full_text"]}),
                    pd.DataFrame({TARGET_NAMES[i]: test_prediction[:, i] for i in range(len(TARGET_NAMES))}),
                ],
                axis=1,
            )

        submission_df_list.append(submission_prediction_df)

    submission_df = pd.concat(submission_df_list)
    submission_df = submission_df.groupby("text_id")[TARGET_NAMES].agg("mean").sort_index().reset_index()

    # save results
    submission_df.to_csv(os.path.join(args.output_dir, "submission.csv"), index=False)


if __name__ == "__main__":
    main()
