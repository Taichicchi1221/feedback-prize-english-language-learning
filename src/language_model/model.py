import torch
import torch.nn as nn

import pytorch_lightning as pl

import transformers

##
from utils import *
from loss import get_loss, get_metric, need_sigmoid, TARGET_MAX, TARGET_MIN
from optimizer import get_optimizer, get_scheduler, AdversarialLearner, hook_sift_layer, AWP


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

    def forward(self, last_hidden_state, hidden_states, attention_mask):
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

    def forward(self, last_hidden_state, hidden_states, attention_mask):
        x = self.layer_norm(last_hidden_state[:, 0, :])
        output = torch.stack([regressor(dropout(x)) for regressor, dropout in zip(self.linears, self.dropouts)]).mean(axis=0)
        return output


class AttentionHead(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(AttentionHead, self).__init__()
        self.W = nn.Linear(in_features, hidden_features)
        self.V = nn.Linear(hidden_features, out_features)

    def forward(self, last_hidden_state, hidden_states, attention_mask):
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

    def forward(self, last_hidden_state, hidden_states, attention_mask):
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

    def forward(self, last_hidden_state, hidden_states, attention_mask):
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

    def forward(self, last_hidden_state, hidden_states, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        norm_mean_embeddings = self.layer_norm(mean_embeddings)
        logits = self.linear(norm_mean_embeddings).squeeze(-1)

        return logits


class MeanPoolingMultiSampleDropoutHead(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        dropout_num,
        dropout_rate,
    ):
        super().__init__()
        self.layer_norm = nn.LayerNorm(in_features)
        self.linear = nn.Linear(in_features, out_features)
        self.dropouts = nn.ModuleList([nn.Dropout(dropout_rate) for _ in range(dropout_num)])
        self.linears = nn.ModuleList([nn.Linear(in_features, out_features) for _ in range(dropout_num)])

    def forward(self, last_hidden_state, hidden_states, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        norm_mean_embeddings = self.layer_norm(mean_embeddings)
        output = torch.stack([regressor(dropout(norm_mean_embeddings)) for regressor, dropout in zip(self.linears, self.dropouts)]).mean(axis=0)
        return output


class MeanMaxPoolingHead(nn.Module):
    def __init__(self, in_features, out_features) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(in_features * 2)
        self.linear = nn.Linear(in_features * 2, out_features)

    def forward(self, last_hidden_state, hidden_states, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask

        max_embeddings, _ = torch.max(last_hidden_state, 1)
        mean_max_embeddings = torch.cat(
            (
                mean_embeddings,
                max_embeddings,
            ),
            1,
        )
        logits = self.linear(self.layer_norm(mean_max_embeddings))

        return logits


class CLSConcatHead(nn.Module):
    def __init__(self, in_features, out_features, num_hidden_layers) -> None:
        super().__init__()
        self.num_hidden_layers = num_hidden_layers
        self.linear = nn.Linear(in_features * num_hidden_layers, out_features)

    def forward(self, last_hidden_state, hidden_states, attention_mask):
        x = torch.cat([hidden_states[-(i + 1)][:, 0, :] for i in range(self.num_hidden_layers)], dim=1)
        output = self.linear(x)
        return output


class WeightedLayerPoolingHead(nn.Module):
    def __init__(self, in_features, out_features, num_hidden_layers, weights=None):
        super().__init__()
        self.num_hidden_layers = num_hidden_layers
        self.layer_weights = (
            nn.Parameter(torch.tensor(weights, dtype=torch.float))
            if weights is not None
            else nn.Parameter(torch.tensor([1] * num_hidden_layers, dtype=torch.float))
        )
        self.layer_norm = nn.LayerNorm(in_features)
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, last_hidden_state, hidden_states, attention_mask):
        all_layer_embedding = torch.stack(hidden_states)
        all_layer_embedding = all_layer_embedding[-self.num_hidden_layers :, :, :, :]

        weight_factor = self.layer_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(all_layer_embedding.size())
        weighted_average = (weight_factor * all_layer_embedding).sum(dim=0) / self.layer_weights.sum()

        input_mask_expanded = attention_mask.unsqueeze(-1).expand(weighted_average.size()).float()
        sum_embeddings = torch.sum(weighted_average * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        norm_mean_embeddings = self.layer_norm(mean_embeddings)

        output = self.linear(norm_mean_embeddings)

        return output


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

    def forward(self, last_hidden_state, hidden_states, attention_mask):
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

    def forward(self, last_hidden_state, hidden_states, attention_mask):
        x, _ = self.gru(last_hidden_state, None)
        x = self.layer_norm(x[:, -1, :])
        output = self.linear(x)
        return output.squeeze(-1)


# ====================================================
# model
# ====================================================
class EvalModel(pl.LightningModule):
    def __init__(
        self,
        config_cfg,
        encoder_cfg,
        head_cfg,
        loss_cfg,
        pretrained=True,
    ):
        super().__init__()

        self.need_sigmoid = need_sigmoid(loss_cfg.type)

        if encoder_cfg.params is None:
            encoder_cfg.params = {}
        if head_cfg.params is None:
            head_cfg.params = {}

        # encoder
        self.config = transformers.AutoConfig.from_pretrained(config_cfg.path)
        self.config.update(encoder_cfg.params)
        if pretrained:
            print("##### model loaded from pretrained #####")
            self.encoder = transformers.AutoModel.from_pretrained(encoder_cfg.path, config=self.config)
        else:
            print("##### model loaded from config #####")
            self.encoder = transformers.AutoModel.from_config(config=self.config)

        # head
        self.head = eval(head_cfg.type)(
            in_features=self.config.hidden_size,
            out_features=6,
            **head_cfg.params,
        )

    def extract_hidden_states_step(self, batch, batch_idx):
        output = self.encoder(
            batch["input_ids"],
            batch["attention_mask"],
            output_hidden_state=True,
        )
        return output["hidden_states"].detach().cpu().float()

    def extract_last_hidden_states_step(self, batch, batch_idx):
        output = self.encoder(
            batch["input_ids"],
            batch["attention_mask"],
            output_hidden_state=True,
        )
        return output["last_hidden_state"].detach().cpu().float()

    def predict_step(self, batch, batch_idx):
        y_hat = self(batch)
        if self.need_sigmoid:
            y_hat = y_hat.sigmoid() * (TARGET_MAX - TARGET_MIN) + TARGET_MIN
        return y_hat.detach().cpu().float()

    def forward(self, x):
        output = self.encoder(
            input_ids=x["input_ids"],
            attention_mask=x["attention_mask"],
            output_hidden_states=True,
        )
        predictions = self.head(
            last_hidden_state=output["last_hidden_state"],
            hidden_states=output["hidden_states"],
            attention_mask=x["attention_mask"],
        )
        return predictions


class Model(EvalModel):
    def __init__(
        self,
        config_cfg,
        encoder_cfg,
        head_cfg,
        loss_cfg,
        metric_cfg,
        optimizer_cfg,
        scheduler_cfg,
        awp_cfg=None,
        sift_cfg=None,
        pretrained=True,
    ):
        super().__init__(
            config_cfg=config_cfg,
            encoder_cfg=encoder_cfg,
            head_cfg=head_cfg,
            loss_cfg=loss_cfg,
            pretrained=pretrained,
        )
        if encoder_cfg.num_freeze_layers:
            self._freeze_encoder(num_freeze_layers=encoder_cfg.num_freeze_layers)
        if encoder_cfg.num_reinit_layers:
            self._reinit_encoder(num_reinit_layers=encoder_cfg.num_reinit_layers)
        if head_cfg.init:
            self._init_head_weights()

        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg

        # awp
        if awp_cfg is not None and awp_cfg.apply:
            self.awp = AWP(**awp_cfg.params)

        # sift
        if sift_cfg is not None and sift_cfg.apply:
            self.sift = AdversarialLearner(self, hook_sift_layer(self, hidden_size=self.encoder.config.hidden_size))

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
                lr_decay_rate=self.optimizer_cfg.lr_decay_rate,
                weight_decay=self.optimizer_cfg.weight_decay,
            ),
            self.optimizer_cfg.type,
            self.optimizer_cfg.params,
        )

        if self.scheduler_cfg.params is None:
            self.scheduler_cfg.params = {}

        scheduler = get_scheduler(
            optimizer,
            self.scheduler_cfg.type,
            self.scheduler_cfg.params,
        )
        if scheduler is None:
            return {"optimizer": optimizer}

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "valid_metric",
                "interval": self.scheduler_cfg.interval,
            },
        }

    def get_optimizer_params(self, encoder_lr, head_lr, lr_decay_rate, weight_decay=0.0):
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

        print("#" * 50)
        print("#", f"layerwise optimization info")

        # head
        print("#", f"head: lr={head_lr:.8f}")
        optimizer_parameters = [
            {
                "params": [p for n, p in self.head.named_parameters()],
                "lr": head_lr,
                "weight_decay": 0.0,
            }
        ]
        # encoder
        ### encoder
        num_layers = self.config.num_hidden_layers
        layers = [self.encoder.embeddings] + [l for l in self.encoder.encoder.layer]
        layers.reverse()

        lr = encoder_lr

        for idx, layer in enumerate(layers):

            # any weights of the layer requires grad?
            requires_grad = any(map(lambda x: x.requires_grad, layer.parameters()))
            if not requires_grad:
                print("#", f"layer{num_layers - idx}: requires_grad: {requires_grad}")
                continue

            if idx == num_layers:
                print("#", f"embeddings: lr={lr:.8f}, weight_decay={weight_decay}, requires_grad: {requires_grad}")
            else:
                print("#", f"layer{num_layers - idx}: lr={lr:.8f}, weight_decay={weight_decay}, requires_grad: {requires_grad}")

            optimizer_parameters += [
                {
                    "params": [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
                    "lr": lr,
                    "weight_decay": weight_decay,
                },
                {
                    "params": [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)],
                    "lr": lr,
                    "weight_decay": 0.0,
                },
            ]
            lr *= lr_decay_rate

        print("#" * 50)

        return optimizer_parameters

    def _init_head_weights(self):
        if self.config.to_dict().get("initializer_range") is None:
            return

        for module in self.head.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)

    def _reinit_encoder(self, num_reinit_layers):
        assert 0 < num_reinit_layers <= self.config.num_hidden_layers, f"num_reinit_layers must be 0 < x <= {self.config.num_hidden_layers}"

        for layer in self.encoder.encoder.layer[-num_reinit_layers:]:
            for module in layer.modules():
                if isinstance(module, nn.Linear):
                    module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                    if module.bias is not None:
                        module.bias.data.zero_()
                elif isinstance(module, nn.Embedding):
                    module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                    if module.padding_idx is not None:
                        module.weight.data[module.padding_idx].zero_()
                elif isinstance(module, nn.LayerNorm):
                    module.bias.data.zero_()
                    module.weight.data.fill_(1.0)

    def _freeze_encoder(self, num_freeze_layers):
        assert 0 < num_freeze_layers <= self.config.num_hidden_layers, f"num_freeze_layers must be 0 < x <= {self.config.num_hidden_layers}"

        # embeddings
        self.encoder.embeddings.requires_grad_(False)

        # encoder
        self.encoder.encoder.layer[:num_freeze_layers].requires_grad_(False)

    def training_step(self, batch, batch_idx):
        y = batch["label"]
        y_hat = self(batch)
        loss = self.criterion(y_hat, y)

        if hasattr(self, "awp"):
            self.awp.attack_backward(self, batch, self.trainer.current_epoch)

        if hasattr(self, "sift"):
            loss = loss + self.sift.loss(y, logits_fn=lambda model, batch: model(batch), batch=batch)

        return {"loss": loss, "y_hat": y_hat, "y": y}

    def on_train_batch_end(self, outputs, batch, batch_idx):
        y = outputs["y"].detach()
        y_hat = outputs["y_hat"].detach()
        if self.need_sigmoid:
            y_hat = y_hat.sigmoid() * (TARGET_MAX - TARGET_MIN) + TARGET_MIN
        self.train_metric(y_hat.detach(), y.detach())
        self.history["lr"].append(self.optimizers(False).param_groups[0]["lr"])

        return super().on_train_batch_end(outputs, batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        y = batch["label"]
        y_hat = self(batch)
        loss = self.criterion(y_hat, y)

        return {"loss": loss, "y_hat": y_hat, "y": y}

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx) -> None:
        y = outputs["y"].detach()
        y_hat = outputs["y_hat"].detach()
        if self.need_sigmoid:
            y_hat = y_hat.sigmoid() * (TARGET_MAX - TARGET_MIN) + TARGET_MIN
        self.valid_metric(y_hat.detach(), y.detach())

        return super().on_validation_batch_end(outputs, batch, batch_idx, dataloader_idx)

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
