from collections import defaultdict

import torch
from lightning import LightningModule
from torch import Tensor, nn
from torch.nn import BCELoss
from torch.optim import AdamW
from torch.optim.optimizer import Optimizer
from torchmetrics.classification import BinaryF1Score, BinaryPrecision, BinaryRecall

from patch_camelyon.modeling.decoder_head import BinaryClassifier
from patch_camelyon.typing import Input, Outputs


class PatchCamelyonModel(LightningModule):
    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.decode_head = BinaryClassifier()
        self.criterion = BCELoss()

        self.metrics = nn.ModuleDict(
            {
                "precision": BinaryPrecision(),
                "recall": BinaryRecall(),
                "f1": BinaryF1Score(),
            }
        )

        self._validation_step_outputs: dict[str, list[Tensor]] = defaultdict(list)

    def forward(self, x: Input) -> Outputs:
        features = self.backbone(x)
        return self.decode_head(features)

    def training_step(self, batch: Input) -> Tensor:
        inputs, targets = batch
        outputs = self(inputs)

        loss = self.criterion(outputs, targets)
        self.log("train/loss", loss, on_step=True, prog_bar=True)

        return loss

    def validation_step(self, batch: Input) -> None:
        inputs, targets = batch
        outputs = self(inputs)

        loss = self.criterion(outputs, targets)
        self._validation_step_outputs["loss"].append(loss)

        for name, metric in self.metrics.items():
            self._validation_step_outputs[name].append(metric(outputs, targets))

    def on_validation_epoch_end(self) -> None:
        avg_outputs = {
            k: torch.stack(v).mean() for k, v in self._validation_step_outputs.items()
        }
        self.log_dict(
            {f"validation/{k}": v for k, v in avg_outputs.items()}, sync_dist=True
        )
        self._validation_step_outputs.clear()

    def test_step(self, batch: Input) -> None:
        inputs, targets = batch
        outputs = self(inputs)

        for metric in self.metrics.values():
            metric(outputs, targets)

    def on_test_epoch_end(self) -> None:
        self.log_dict(
            {name: metric.compute() for name, metric in self.metrics.items()},
            sync_dist=True,
        )

        for metric in self.metrics.values():
            metric.reset()

    def configure_optimizers(self) -> Optimizer:
        return AdamW(self.parameters(), lr=1e-4)
