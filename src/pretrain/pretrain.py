import os
import random
import glob
import shutil

import numpy as np
import pandas as pd

import torch

import transformers


def seed_everything(seed: int = 42, deterministic: bool = False):
    """Set seeds"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = deterministic


def prepare_data(num_samples=None):
    FEEDBACK1_DIR = "../input/feedback-prize-2021"
    FEEDBACK2_DIR = "../input/feedback-prize-effectiveness"
    FEEDBACK3_DIR = "../input/feedback-prize-english-language-learning"

    # feedback1
    df1 = pd.read_csv(os.path.join(FEEDBACK1_DIR, "train.csv"))
    feedback1_ids = set(df1["id"])

    # feedback2
    df2 = pd.read_csv(os.path.join(FEEDBACK2_DIR, "train.csv"))
    feedback2_ids = set(df2["essay_id"])

    # feedback3
    df3 = pd.read_csv(os.path.join(FEEDBACK3_DIR, "train.csv"))
    feedback3_ids = set(df3["text_id"])

    target_ids = (feedback1_ids | feedback2_ids) - feedback3_ids
    data = {
        "id": [],
        "text": [],
    }

    for text_id in target_ids:
        if text_id in feedback3_ids:
            continue
        elif text_id in feedback1_ids:
            path = os.path.join(FEEDBACK1_DIR, "train", f"{text_id}.txt")
            with open(path) as f:
                text = f.read()
        elif text_id in feedback2_ids:
            path = os.path.join(FEEDBACK2_DIR, "train", f"{text_id}.txt")
            with open(path) as f:
                text = f.read()

        data["id"].append(text_id)
        data["text"].append(text)

    # debug?
    if num_samples:
        data["id"] = data["id"][:1000]
        data["text"] = data["text"][:1000]

    return data


class Dataset:
    def __init__(self, ids, texts, tokenizer, max_length):
        self.ids = ids
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, item):
        encoded = self.tokenizer.encode_plus(
            self.texts[item],
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            return_special_tokens_mask=True,
        )
        return encoded

    def __len__(self):
        return len(self.ids)


def save_pretrain_results(transformers_model_path):
    DIR = "../input/pretrained_models"
    transformers_model_path_base = os.path.split(transformers_model_path)[-1]
    for exist_path in glob.glob(os.path.join(DIR, f"{transformers_model_path_base}*")):
        if os.path.exists(exist_path):
            shutil.rmtree(exist_path)

    checkpoint_paths = sorted(glob.glob("checkpoint*"), key=lambda x: int(x.split("-")[-1]))
    idx = 1
    for checkpoint_path in checkpoint_paths:
        print(f"### idx: {idx}, checkpoint_path: {checkpoint_path}")
        tokenizer = transformers.AutoTokenizer.from_pretrained(transformers_model_path)
        tokenizer.save_pretrained(os.path.join(DIR, f"{transformers_model_path_base}_{idx}"))
        model = transformers.AutoModel.from_pretrained(checkpoint_path)
        model.save_pretrained(os.path.join(DIR, f"{transformers_model_path_base}_{idx}"))
        idx += 1


def main(cfg):
    num_samples = None
    if cfg.globals.debug:
        num_samples = 1000
        cfg.tokenizer.max_length = 32

    seed_everything(seed=cfg.globals.seed)

    data = prepare_data(num_samples=num_samples)

    # tokenizer, model
    if cfg.tokenizer.params is None:
        cfg.tokenizer.params = {}
    if cfg.model.params is None:
        cfg.model.params = {}
    transformers_config = transformers.AutoConfig.from_pretrained(cfg.tokenizer.path)
    transformers_config.update(cfg.tokenizer.params)
    transformers_config.update(cfg.model.params)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        cfg.tokenizer.path,
        config=transformers_config,
    )
    model = transformers.AutoModelForMaskedLM.from_pretrained(
        cfg.model.path,
        config=transformers_config,
    )
    if cfg.model.freeze_embeddings:
        if "deberta" in cfg.model.path:
            model.deberta.embeddings.requires_grad_(False)
        elif "bert" in cfg.model.path:
            model.bert.embeddings.requires_grad_(False)
        elif "longformer" in cfg.model.path:
            model.longformer.embeddings.requires_grad_(False)
        elif "bigbird" in cfg.model.path:
            model.bert.embeddings.requires_grad_(False)
    if cfg.model.freeze_encoders:
        if "deberta" in cfg.model.path:
            model.deberta.encoder.layer[: cfg.model.freeze_encoders].requires_grad_(False)
        elif "bert" in cfg.model.path:
            model.bert.encoder.layer[: cfg.model.freeze_encoders].requires_grad_(False)
        elif "longformer" in cfg.model.path:
            model.longformer.encoder.layer[: cfg.model.freeze_encoders].requires_grad_(False)
        elif "bigbird" in cfg.model.path:
            model.bert.encoder.layer[: cfg.model.freeze_encoders].requires_grad_(False)

    # dataset
    train_dataset = Dataset(
        ids=data["id"],
        texts=data["text"],
        tokenizer=tokenizer,
        max_length=cfg.tokenizer.max_length,
    )
    valid_dataset = Dataset(
        ids=data["id"],
        texts=data["text"],
        tokenizer=tokenizer,
        max_length=cfg.tokenizer.max_length,
    )

    data_collator = transformers.DataCollatorForLanguageModeling(tokenizer=tokenizer, **cfg.collator)

    training_args = transformers.TrainingArguments(**cfg.training)

    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
    )
    trainer.train()

    # save results
    if not cfg.globals.debug:
        save_pretrain_results(transformers_model_path=cfg.model.path)
