__all__ = ["ModelAdapter"]

from icevision.imports import *
from icevision.metrics import *
from icevision.engines.lightning.lightning_model_adapter import LightningModelAdapter
from torch.nn import CrossEntropyLoss
from icevision.models.fastai import unet


class ModelAdapter(LightningModelAdapter, ABC):
    """Lightning module specialized for EfficientDet, with metrics support.

    The methods `forward`, `training_step`, `validation_step`, `validation_epoch_end`
    are already overriden.

    # Arguments
        model: The pytorch model to use.
        metrics: `Sequence` of metrics to use.

    # Returns
        A `LightningModule`.
    """

    def __init__(self, model: nn.Module, metrics: List[Metric] = None):
        super().__init__(metrics=metrics)
        self.model = model
        self.loss_func = CrossEntropyLoss()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        (xb, yb), _ = batch
        preds = self(xb)

        loss = self.loss_func(preds, yb)

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        (xb, yb), records = batch

        with torch.no_grad():
            preds = self(xb)
            loss = self.loss_func(preds, yb)

            preds = unet.convert_raw_predictions(
                batch=xb,
                raw_preds=preds,
                records=records,
            )

        self.accumulate_metrics(preds)

        self.log("val_loss", loss)

    def validation_epoch_end(self, outs):
        self.finalize_metrics()
