import numpy as np
import torch

# ====================================================
# dataset
# ====================================================
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
        self.texts = df["full_text"].to_numpy()
        self.labels = df[target_names].to_numpy()

        self.length = len(df)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        text = self.texts[idx]

        output = self.tokenizer.encode_plus(
            text,
            max_length=self.max_length,
            truncation=True,
        )

        output["text_id"] = self.text_ids[idx]
        output["label"] = self.labels[idx]

        return output


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
