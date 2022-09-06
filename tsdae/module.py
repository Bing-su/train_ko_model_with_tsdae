from __future__ import annotations

import pytorch_lightning as pl
import torch
from pytorch_optimizer import load_optimizer
from sentence_transformers import SentenceTransformer
from sentence_transformers.losses import DenoisingAutoEncoderLoss

from .util import build_sentence_transformer


class KoTSDAEModule(pl.LightningModule):
    def __init__(
        self,
        model: str | SentenceTransformer,
        optimizer_name: str = "adamp",
        lr: float = 5e-5,
        weight_decay: float = 0.0,
        decoder_name: str | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        if isinstance(model, str):
            self.model = build_sentence_transformer(model)
        else:
            self.model = model

        self.optimizer_name = optimizer_name.lower()
        self.lr = lr
        self.weight_decay = weight_decay

        if not decoder_name:  # default
            self.loss = DenoisingAutoEncoderLoss(self.model, tie_encoder_decoder=True)
        else:
            self.loss = DenoisingAutoEncoderLoss(
                self.model, decoder_name, tie_encoder_decoder=False
            )

    def configure_optimizers(self):
        param_optimizer = list(self.model.named_parameters())

        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        opt_class = load_optimizer(self.optimizer_name)
        optimizer = opt_class(
            optimizer_grouped_parameters, lr=self.lr, weight_decay=self.weight_decay
        )

        cycle_momentum = self.optimizer_name not in ("adan", "adapnm")
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            0.01,
            total_steps=self.trainer.estimated_stepping_batches,
            cycle_momentum=cycle_momentum,
        )

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        features, labels = batch
        loss = self.loss(features, labels)

        self.log(
            "train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss
