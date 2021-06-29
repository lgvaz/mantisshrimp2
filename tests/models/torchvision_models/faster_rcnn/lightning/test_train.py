import pytest
from icevision.all import *
from icevision.models.torchvision import faster_rcnn


@pytest.fixture
def light_model_cls():
    class LightModel(faster_rcnn.lightning.ModelAdapter):
        def configure_optimizers(self):
            return SGD(self.parameters(), lr=1e-3)

    return LightModel


@pytest.mark.parametrize("metrics", [[], [COCOMetric()]])
def test_lightining_faster_rcnn_train(
    fridge_faster_rcnn_dls, fridge_faster_rcnn_model, light_model_cls, metrics
):
    train_dl, valid_dl = fridge_faster_rcnn_dls
    light_model = light_model_cls(fridge_faster_rcnn_model, metrics=metrics)

    trainer = pl.Trainer(
        max_epochs=1,
        weights_summary=None,
        num_sanity_val_steps=0,
        logger=False,
        checkpoint_callback=False,
    )
    trainer.fit(light_model, train_dl, valid_dl)
