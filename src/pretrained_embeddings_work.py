import os
import sys
import re
import gc
import glob
import shutil

from tqdm.auto import tqdm

import json

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

import transformers

TARGET_NAMES = [
    "cohesion",
    "syntax",
    "vocabulary",
    "phraseology",
    "grammar",
    "conventions",
]


DATASET_DIR = "../datasets"
INPUT_DIR = "../input/feedback-prize-english-language-learning"
PRETRAINED_MODEL_OUTPUT_DIR = "../datasets/pretrained-models"
EMBEDDINGS_OUTPUT_DIR = "../datasets/feedback3-pretrained-embeddings"

CONFIG_LIST = [
    {"model_path": "microsoft/deberta-v3-base", "max_length": 512, "pool": "mean", "batch_size": 32},
    {"model_path": "microsoft/deberta-v3-base", "max_length": 768, "pool": "mean", "batch_size": 32},
    {"model_path": "microsoft/deberta-v3-base", "max_length": 1024, "pool": "mean", "batch_size": 16},
    {"model_path": "microsoft/deberta-v3-large", "max_length": 512, "pool": "mean", "batch_size": 16},
    {"model_path": "microsoft/deberta-v3-large", "max_length": 768, "pool": "mean", "batch_size": 16},
    {"model_path": "microsoft/deberta-v3-large", "max_length": 1024, "pool": "mean", "batch_size": 8},
    {"model_path": "microsoft/deberta-xlarge", "max_length": 512, "pool": "mean", "batch_size": 16},
    {"model_path": "microsoft/deberta-xlarge", "max_length": 768, "pool": "mean", "batch_size": 16},
    {"model_path": "microsoft/deberta-xlarge", "max_length": 1024, "pool": "mean", "batch_size": 8},
    {"model_path": "microsoft/deberta-v2-xlarge", "max_length": 512, "pool": "mean", "batch_size": 16},
    {"model_path": "microsoft/deberta-v2-xlarge", "max_length": 768, "pool": "mean", "batch_size": 16},
    {"model_path": "microsoft/deberta-v2-xlarge", "max_length": 1024, "pool": "mean", "batch_size": 8},
    {"model_path": "microsoft/deberta-large-mnli", "max_length": 512, "pool": "mean", "batch_size": 16},
    {"model_path": "microsoft/deberta-large-mnli", "max_length": 768, "pool": "mean", "batch_size": 16},
    {"model_path": "microsoft/deberta-large-mnli", "max_length": 1024, "pool": "mean", "batch_size": 8},
    {"model_path": "facebook/bart-base", "max_length": 512, "pool": "mean", "batch_size": 32},
    {"model_path": "facebook/bart-base", "max_length": 768, "pool": "mean", "batch_size": 16},
    {"model_path": "facebook/bart-base", "max_length": 1024, "pool": "mean", "batch_size": 8},
    {"model_path": "facebook/bart-large", "max_length": 512, "pool": "mean", "batch_size": 16},
    {"model_path": "facebook/bart-large", "max_length": 768, "pool": "mean", "batch_size": 8},
    {"model_path": "facebook/bart-large", "max_length": 1024, "pool": "mean", "batch_size": 8},
    {"model_path": "roberta-base", "max_length": 512, "pool": "mean", "batch_size": 32},
    {"model_path": "roberta-large", "max_length": 512, "pool": "mean", "batch_size": 16},
    {"model_path": "facebook/muppet-roberta-base", "max_length": 512, "pool": "mean", "batch_size": 32},
    {"model_path": "facebook/muppet-roberta-large", "max_length": 512, "pool": "mean", "batch_size": 16},
    {"model_path": "google/electra-large-discriminator", "max_length": 512, "pool": "mean", "batch_size": 16},
    {"model_path": "allenai/longformer-base-4096", "max_length": 4096, "pool": "mean", "batch_size": 8},
    {"model_path": "allenai/longformer-large-4096", "max_length": 4096, "pool": "mean", "batch_size": 8},
    {"model_path": "google/bigbird-roberta-base", "max_length": 4096, "pool": "mean", "batch_size": 8},
    {"model_path": "google/bigbird-roberta-large", "max_length": 4096, "pool": "mean", "batch_size": 8},
    {"model_path": "microsoft/deberta-v3-base", "max_length": 512, "pool": "CLS", "batch_size": 32},
    {"model_path": "microsoft/deberta-v3-base", "max_length": 768, "pool": "CLS", "batch_size": 32},
    {"model_path": "microsoft/deberta-v3-base", "max_length": 1024, "pool": "CLS", "batch_size": 16},
    {"model_path": "microsoft/deberta-v3-large", "max_length": 512, "pool": "CLS", "batch_size": 16},
    {"model_path": "microsoft/deberta-v3-large", "max_length": 768, "pool": "CLS", "batch_size": 16},
    {"model_path": "microsoft/deberta-v3-large", "max_length": 1024, "pool": "CLS", "batch_size": 8},
    {"model_path": "microsoft/deberta-xlarge", "max_length": 512, "pool": "CLS", "batch_size": 16},
    {"model_path": "microsoft/deberta-xlarge", "max_length": 768, "pool": "CLS", "batch_size": 16},
    {"model_path": "microsoft/deberta-xlarge", "max_length": 1024, "pool": "CLS", "batch_size": 8},
    {"model_path": "microsoft/deberta-v2-xlarge", "max_length": 512, "pool": "CLS", "batch_size": 16},
    {"model_path": "microsoft/deberta-v2-xlarge", "max_length": 768, "pool": "CLS", "batch_size": 16},
    {"model_path": "microsoft/deberta-v2-xlarge", "max_length": 1024, "pool": "CLS", "batch_size": 8},
    {"model_path": "microsoft/deberta-large-mnli", "max_length": 512, "pool": "CLS", "batch_size": 16},
    {"model_path": "microsoft/deberta-large-mnli", "max_length": 768, "pool": "CLS", "batch_size": 16},
    {"model_path": "microsoft/deberta-large-mnli", "max_length": 1024, "pool": "CLS", "batch_size": 8},
    {"model_path": "facebook/bart-base", "max_length": 512, "pool": "CLS", "batch_size": 32},
    {"model_path": "facebook/bart-base", "max_length": 768, "pool": "CLS", "batch_size": 16},
    {"model_path": "facebook/bart-base", "max_length": 1024, "pool": "CLS", "batch_size": 8},
    {"model_path": "facebook/bart-large", "max_length": 512, "pool": "CLS", "batch_size": 16},
    {"model_path": "facebook/bart-large", "max_length": 768, "pool": "CLS", "batch_size": 8},
    {"model_path": "facebook/bart-large", "max_length": 1024, "pool": "CLS", "batch_size": 8},
    {"model_path": "roberta-base", "max_length": 512, "pool": "CLS", "batch_size": 32},
    {"model_path": "roberta-large", "max_length": 512, "pool": "CLS", "batch_size": 16},
    {"model_path": "facebook/muppet-roberta-base", "max_length": 512, "pool": "CLS", "batch_size": 32},
    {"model_path": "facebook/muppet-roberta-large", "max_length": 512, "pool": "CLS", "batch_size": 16},
    {"model_path": "google/electra-large-discriminator", "max_length": 512, "pool": "CLS", "batch_size": 16},
    {"model_path": "allenai/longformer-base-4096", "max_length": 4096, "pool": "CLS", "batch_size": 8},
    {"model_path": "allenai/longformer-large-4096", "max_length": 4096, "pool": "CLS", "batch_size": 8},
    {"model_path": "google/bigbird-roberta-base", "max_length": 4096, "pool": "CLS", "batch_size": 8},
    {"model_path": "google/bigbird-roberta-large", "max_length": 4096, "pool": "CLS", "batch_size": 8},
]


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df=None,
        tokenizer=None,
        max_length=None,
        target_names=None,
    ):
        assert max_length is not None
        self.max_length = max_length

        self.tokenizer = tokenizer

        self.text_ids = df["text_id"].to_numpy()

        self.input_ids = []
        self.attention_mask = []

        for text in df["full_text"].to_list():
            tokens = self.tokenizer.encode_plus(
                text.rstrip(),
                truncation=True,
                max_length=self.max_length,
            )
            self.input_ids.append(tokens["input_ids"])
            self.attention_mask.append(tokens["attention_mask"])

        self.texts = df["full_text"].to_numpy()
        self.labels = df[target_names].to_numpy()

        self.length = len(df)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "text_id": self.text_ids[idx],
            "text": self.texts[idx],
            "label": self.labels[idx],
        }


class Collate:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        output = dict()

        output["input_ids"] = [sample["input_ids"] for sample in batch]
        output["attention_mask"] = [sample["attention_mask"] for sample in batch]

        # calculate max token length of this batch
        batch_max = max([len(input_ids) for input_ids in output["input_ids"]])

        # add padding
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id

        if self.tokenizer.padding_side == "right":
            output["input_ids"] = [s + (batch_max - len(s)) * [pad_token_id] for s in output["input_ids"]]
            output["attention_mask"] = [s + (batch_max - len(s)) * [0] for s in output["attention_mask"]]
        else:
            output["input_ids"] = [(batch_max - len(s)) * [pad_token_id] + s for s in output["input_ids"]]
            output["attention_mask"] = [(batch_max - len(s)) * [0] + s for s in output["attention_mask"]]

        # convert to tensors
        output["input_ids"] = torch.tensor(
            output["input_ids"],
            dtype=torch.long,
        )
        output["attention_mask"] = torch.tensor(
            output["attention_mask"],
            dtype=torch.long,
        )

        output["label"] = torch.tensor(
            np.stack([sample["label"] for sample in batch], axis=0),
            dtype=torch.float32,
        )

        return output


def cls_pooling(model_output, attention_mask):
    return model_output.last_hidden_state.detach().cpu()[:, 0, :]


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state.detach().cpu()
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def get_embeddings(df, model_path=None, max_length=None, pool_method="CLS", batch_size=1, DEVICE="cuda"):
    assert model_path is not None
    assert max_length is not None

    model = transformers.AutoModel.from_pretrained(model_path, local_files_only=True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, local_files_only=True)

    dataset = Dataset(df, tokenizer=tokenizer, max_length=max_length, target_names=TARGET_NAMES)
    collate_function = Collate(tokenizer, max_length=max_length)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collate_function,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    model = model.to(DEVICE)
    model.eval()
    text_features = []
    for batch in tqdm(dataloader, total=len(dataloader)):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        with torch.no_grad():
            model_output = model(input_ids=input_ids, attention_mask=attention_mask)

        if pool_method == "CLS":
            sentence_embeddings = cls_pooling(model_output, attention_mask.detach().cpu())
        if pool_method == "mean":
            sentence_embeddings = mean_pooling(model_output, attention_mask.detach().cpu())
        # Normalize the embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        sentence_embeddings = sentence_embeddings.squeeze(0).detach().cpu().numpy()
        text_features.append(sentence_embeddings)
    text_features = np.concatenate(text_features, axis=0)
    print("embeddings shape", text_features.shape)

    return text_features


def save_pretrained_config_model(model_path):
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    model = transformers.AutoModel.from_pretrained(model_path)
    tokenizer.save_pretrained(os.path.join(PRETRAINED_MODEL_OUTPUT_DIR, os.path.basename(model_path)))
    model.save_pretrained(os.path.join(PRETRAINED_MODEL_OUTPUT_DIR, os.path.basename(model_path)))
    print(f"save pretrained: {model_path}")


def save_pretrained_embeddings_main():
    df = pd.read_csv("../input/feedback-prize-english-language-learning/train.csv")
    print(f"Train shape: {df.shape}")

    for idx, config in enumerate(CONFIG_LIST):
        print(f"{idx + 1} / {len(CONFIG_LIST)}", config)
        model_base_name = os.path.basename(config["model_path"])

        save_pretrained_path = os.path.join(PRETRAINED_MODEL_OUTPUT_DIR, model_base_name)
        if not os.path.exists(save_pretrained_path):
            save_pretrained_config_model(model_path=config["model_path"])

        save_embeddings_path = os.path.join(EMBEDDINGS_OUTPUT_DIR, f"{model_base_name}_{config['max_length']}_{config['pool']}.npy")
        if not os.path.exists(save_embeddings_path):
            text_features = get_embeddings(
                df,
                model_path=save_pretrained_path,
                max_length=config["max_length"],
                batch_size=config["batch_size"],
                pool_method=config["pool"],
            )
            np.save(
                save_embeddings_path,
                text_features,
                allow_pickle=True,
            )

            print(f"text_features={text_features.shape}")


if __name__ == "__main__":
    shutil.copy(__file__, EMBEDDINGS_OUTPUT_DIR)
    save_pretrained_embeddings_main()
