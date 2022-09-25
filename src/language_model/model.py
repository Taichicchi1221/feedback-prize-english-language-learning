import torch
import torch.nn as nn

from omegaconf import OmegaConf

import pytorch_lightning as pl

import transformers

##
from utils import *
from loss import get_loss, get_metric
from optimizer import get_optimizer, get_scheduler

# ====================================================
# heads
# ====================================================
class SimpleHead(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        dropout_rate,
    ):
        super().__init__()
        self.layer_norm = nn.LayerNorm(in_features)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(in_features=in_features, out_features=out_features)

    def forward(self, last_hidden_state, attention_mask):
        x = self.layer_norm(last_hidden_state[:, 0, :])
        x = self.dropout(x)
        output = self.linear(x)
        return output


class MultiSampleDropoutHead(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        dropout_num,
        dropout_rate,
    ):
        super().__init__()
        self.layer_norm = nn.LayerNorm(in_features)
        self.dropouts = nn.ModuleList([nn.Dropout(dropout_rate) for _ in range(dropout_num)])
        self.linears = nn.ModuleList([nn.Linear(in_features, out_features) for _ in range(dropout_num)])

    def forward(self, last_hidden_state, attention_mask):
        x = self.layer_norm(last_hidden_state[:, 0, :])
        output = torch.stack([regressor(dropout(x)) for regressor, dropout in zip(self.linears, self.dropouts)]).mean(axis=0)
        return output


class AttentionHead(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(AttentionHead, self).__init__()
        self.W = nn.Linear(in_features, hidden_features)
        self.V = nn.Linear(hidden_features, out_features)

    def forward(self, last_hidden_state, attention_mask):
        attention_scores = self.V(torch.tanh(self.W(last_hidden_state)))
        attention_scores = torch.softmax(attention_scores, dim=1)
        attentive_x = attention_scores * last_hidden_state
        attentive_x = attentive_x.sum(axis=1)
        return attentive_x


class MaskAddedAttentionHead(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(MaskAddedAttentionHead, self).__init__()
        self.W = nn.Linear(in_features, hidden_features)
        self.V = nn.Linear(hidden_features, out_features)

    def forward(self, last_hidden_state, attention_mask):
        attention_scores = self.V(torch.tanh(self.W(last_hidden_state)))
        attention_scores = attention_scores + attention_mask
        attention_scores = torch.softmax(attention_scores, dim=1)
        attentive_x = attention_scores * last_hidden_state
        attentive_x = attentive_x.sum(axis=1)
        return attentive_x


class AttentionPoolHead(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.LayerNorm(in_features),
            nn.GELU(),
            nn.Linear(in_features, 1),
        )
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, last_hidden_state, attention_mask):
        w = self.attention(last_hidden_state).float()
        w[attention_mask == 0] = float("-inf")
        w = torch.softmax(w, 1)
        x = torch.sum(w * last_hidden_state, dim=1)
        output = self.linear(x)
        return output


class MeanPoolingHead(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.layer_norm = nn.LayerNorm(in_features)
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        norm_mean_embeddings = self.layer_norm(mean_embeddings)
        logits = self.linear(norm_mean_embeddings).squeeze(-1)

        return logits


class MeanMaxPoolingHead(nn.Module):
    def __init__(self, in_features, out_features) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features * 2, out_features)

    def forward(self, last_hidden_state, attention_mask):
        mean_pooling_embeddings = torch.mean(last_hidden_state, 1)
        max_pooling_embeddings, _ = torch.max(last_hidden_state, 1)
        mean_max_embeddings = torch.cat(
            (
                mean_pooling_embeddings,
                max_pooling_embeddings,
            ),
            1,
        )
        logits = self.linear(mean_max_embeddings)  # twice the hidden size

        return logits


class LSTMHead(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        dropout,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_features,
            hidden_size=in_features,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.layer_norm = nn.LayerNorm(in_features * 2)
        self.linear = nn.Linear(in_features * 2, out_features)

    def forward(self, last_hidden_state, attention_mask):
        x, _ = self.lstm(last_hidden_state, None)
        x = self.layer_norm(x[:, -1, :])
        output = self.linear(x)
        return output


class GRUHead(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        dropout,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=in_features,
            hidden_size=in_features,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.layer_norm = nn.LayerNorm(in_features * 2)
        self.linear = nn.Linear(in_features * 2, out_features)

    def forward(self, last_hidden_state, attention_mask):
        x, _ = self.gru(last_hidden_state, None)
        x = self.layer_norm(x[:, -1, :])
        output = self.linear(x)
        return output.squeeze(-1)


class CNNHead(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        hidden_size,
        kernel_size,
    ):
        super().__init__()
        self.cnn1 = nn.Conv1d(in_features, hidden_size, kernel_size=kernel_size, padding=1)
        self.cnn2 = nn.Conv1d(hidden_size, out_features, kernel_size=kernel_size, padding=1)
        self.activation = nn.PReLU()

    def forward(self, last_hidden_state, attention_mask):
        x = last_hidden_state.permute(0, 2, 1)
        x = self.cnn1(x)
        x = self.activation(x)
        x = self.cnn2(x)
        x, _ = torch.max(x, 2)
        return x


# ====================================================
# model
# ====================================================
class EvalModel(pl.LightningModule):
    def __init__(
        self,
        encoder_cfg,
        head_cfg,
        transformers_config=None,
    ):
        super().__init__()

        if encoder_cfg.params is None:
            encoder_cfg.params = {}
        if head_cfg.params is None:
            head_cfg.params = {}

        # encoder
        if transformers_config is None:
            self.config = transformers.AutoConfig.from_pretrained(encoder_cfg.path)
            self.config.update(encoder_cfg.params)
            self.encoder = transformers.AutoModel.from_pretrained(
                encoder_cfg.path,
                config=self.config,
            )
        else:
            self.config = transformers_config
            self.config.update(encoder_cfg.params)
            self.encoder = transformers.AutoModel.from_config(self.config)

        # head
        self.head = eval(head_cfg.type)(
            in_features=self.config.hidden_size,
            out_features=6,
            **head_cfg.params,
        )

    def predict_step(self, batch, batch_idx):
        y_hat = self(batch)
        return y_hat.detach().cpu().float()

    def forward(self, x):
        last_hidden_state = self.encoder(
            input_ids=x["input_ids"],
            attention_mask=x["attention_mask"],
        )["last_hidden_state"]
        predictions = self.head(last_hidden_state, x["attention_mask"])
        return predictions


class Model(EvalModel):
    def __init__(
        self,
        encoder_cfg,
        head_cfg,
        loss_cfg,
        metric_cfg,
        optimizer_cfg,
        scheduler_cfg,
        transformers_config=None,
    ):
        super().__init__(encoder_cfg=encoder_cfg, head_cfg=head_cfg, transformers_config=transformers_config)
        for module in self.head.modules():
            self._init_weights(module, self.config)

        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg

        self.criterion = get_loss(loss_cfg.type, loss_cfg.params)

        self.train_metric = get_metric(metric_cfg.type, metric_cfg.params)
        self.valid_metric = get_metric(metric_cfg.type, metric_cfg.params)

        # init model training histories
        self.history = {
            "train_metric": [],
            "valid_metric": [],
            "lr": [],
        }

    def configure_optimizers(self):
        optimizer = get_optimizer(
            self.get_optimizer_params(
                encoder_lr=self.optimizer_cfg.lr.encoder,
                head_lr=self.optimizer_cfg.lr.head,
                weight_decay=self.optimizer_cfg.weight_decay,
            ),
            self.optimizer_cfg.type,
            self.optimizer_cfg.params,
        )

        scheduler = get_scheduler(
            optimizer,
            self.scheduler_cfg.type,
            OmegaConf.to_object(self.scheduler_cfg.params),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "valid_metric",
                "interval": self.scheduler_cfg.interval,
            },
        }

    def get_optimizer_params(self, encoder_lr, head_lr, weight_decay=0.0):
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {
                "params": [p for n, p in self.encoder.named_parameters() if not any(nd in n for nd in no_decay)],
                "lr": encoder_lr,
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in self.encoder.named_parameters() if any(nd in n for nd in no_decay)],
                "lr": encoder_lr,
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in self.head.named_parameters()],
                "lr": head_lr,
                "weight_decay": 0.0,
            },
        ]
        return optimizer_parameters

    def _init_weights(self, module, config):
        if config.to_dict().get("initializer_range") is None:
            return
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def training_step(self, batch, batch_idx):
        y = batch["label"]
        y_hat = self(batch)
        loss = self.criterion(y_hat, y)

        self.train_metric(y_hat.detach(), y.detach())
        self.history["lr"].append(self.optimizers(False).param_groups[0]["lr"])

        return loss

    def validation_step(self, batch, batch_idx):
        y = batch["label"]
        y_hat = self(batch)
        loss = self.criterion(y_hat, y)

        self.valid_metric(y_hat.detach(), y.detach())

        return loss

    def on_validation_epoch_end(self) -> None:
        self.log(
            name="train_metric",
            value=self.train_metric,
            prog_bar=True,
            logger=False,
            on_step=False,
            on_epoch=True,
        )
        self.history["train_metric"].append(self.train_metric.compute().detach().cpu().numpy())
        self.log(
            name="valid_metric",
            value=self.valid_metric,
            prog_bar=True,
            logger=False,
            on_step=False,
            on_epoch=True,
        )
        self.history["valid_metric"].append(self.valid_metric.compute().detach().cpu().numpy())
        return super().on_validation_epoch_end()
