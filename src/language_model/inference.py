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
def main():
    cfg = OmegaConf.load("config.yaml")
    transformers_config = torch.load("transformers_config.pth")
    model_path_list = joblib.load("model_path_list.pkl")

    seed_everything(cfg.globals.seed, deterministic=True)

    test_df = pd.read_csv("../input/feedback-prize-english-language-learning/test.csv")
    for target_name in TARGET_NAMES:
        test_df[target_name] = -1

    submission_df_list = []
    for fold, model_path in enumerate(model_path_list):
        tokenizer = get_tokenizer(
            tokenizer_path="tokenizer",
            tokenizer_params=cfg.tokenizer.params,
            transformers_config=transformers_config,
        )

        print("#" * 30, f"model_path: {model_path}", "#" * 30)
        model = EvalModel.load_from_checkpoint(
            model_path,
            encoder_cfg=cfg.model.encoder,
            head_cfg=cfg.model.head,
            transformers_config=transformers_config,
        )

        predict_trainer = pl.Trainer(**cfg.predict_trainer)

        test_dataset = Dataset(df=test_df, tokenizer=tokenizer, max_length=cfg.tokenizer.max_length.test, target_names=TARGET_NAMES)
        test_collate_fn = Collate(tokenizer, max_length=cfg.tokenizer.max_length.test)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, collate_fn=test_collate_fn, **cfg.dataloader.test)

        submission_prediction_df = pd.DataFrame(
            torch.cat(predict_trainer.predict(model=model, dataloaders=test_dataloader)).numpy(),
            columns=TARGET_NAMES,
        )
        submission_prediction_df.insert(0, "text_id", test_dataset.text_ids)
        submission_df_list.append(submission_prediction_df)

    submission_df = pd.concat(submission_df_list)
    submission_df = submission_df.groupby("text_id").agg("mean").sort_index().reset_index()

    # save results
    submission_df.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()
