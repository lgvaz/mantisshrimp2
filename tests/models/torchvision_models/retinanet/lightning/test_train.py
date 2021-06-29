import pytest
from icevision.all import *
from icevision.models.torchvision import retinanet


@pytest.fixture
def light_model_cls():
    class LightModel(retinanet.lightning.ModelAdapter):
        def configure_optimizers(self):
            return SGD(self.parameters(), lr=1e-4)

    return LightModel


@pytest.mark.parametrize("metrics", [[], [COCOMetric()]])
def test_lightining_retinanet_train(
    fridge_faster_rcnn_dls, fridge_retinanet_model, light_model_cls, metrics
):
    train_dl, valid_dl = fridge_faster_rcnn_dls
    light_model = light_model_cls(fridge_retinanet_model, metrics=metrics)

    trainer = pl.Trainer(
        max_epochs=1,
        weights_summary=None,
        num_sanity_val_steps=0,
        logger=False,
        checkpoint_callback=False,
    )
    trainer.fit(light_model, train_dl, valid_dl)
