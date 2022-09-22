from typing import Optional, Union

import pytorch_lightning as pl
import torch
from loguru import logger
from sentence_transformers import SentenceTransformer
from sentence_transformers.losses import DenoisingAutoEncoderLoss

from .util import build_sentence_transformer, create_optimizer


class KoTSDAEModule(pl.LightningModule):
    def __init__(
        self,
        model: Union[str, SentenceTransformer],
        optimizer_name: str = "adamw",
        lr: float = 5e-5,
        weight_decay: float = 1e-5,
        max_lr: float = 1e-3,
        decoder_name: Optional[str] = None,
        max_seq_length: Optional[int] = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        if isinstance(model, str):
            self.model = build_sentence_transformer(model, max_seq_length)
        else:
            self.model = model

        self.optimizer_name = optimizer_name.lower()
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_lr = max_lr

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
        opt_class = create_optimizer(self.optimizer_name)
        optimizer = opt_class(optimizer_grouped_parameters, lr=self.lr)

        # bnb Embedding 설정
        if "bnb" in self.optimizer_name:
            from bitsandbytes.optim import GlobalOptimManager

            manager = GlobalOptimManager.get_instance()
            for module in self.model.modules():
                if isinstance(module, torch.nn.Embedding):
                    manager.register_module_override(
                        module, "weight", {"optim_bits": 32}
                    )
                    logger.debug(f"bitsandbytes: will optimize {module} in fp32")

        # scheduler 설정
        cycle_momentum = self.optimizer_name not in ("adan", "adapnm")
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            self.max_lr,
            total_steps=self.trainer.estimated_stepping_batches,
            cycle_momentum=cycle_momentum,
            div_factor=10.0,
        )
        scheduler_config = {"scheduler": scheduler, "interval": "step"}

        return [optimizer], [scheduler_config]

    def training_step(self, batch, batch_idx):
        features, labels = batch
        loss = self.loss(features, labels)
        self.log("train/loss", loss, on_step=True, on_epoch=True)
        return loss

    @property
    def save(self):
        return self.model.save
