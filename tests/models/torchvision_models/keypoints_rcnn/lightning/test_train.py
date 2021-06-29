import pytest
from icevision.all import *
from icevision.models.torchvision import keypoint_rcnn


@pytest.fixture
def light_model_cls():
    class LightModel(keypoint_rcnn.lightning.ModelAdapter):
        def configure_optimizers(self):
            return SGD(self.parameters(), lr=1e-4)

    return LightModel


def test_lightining_keypoints_rcnn_train(ochuman_keypoints_dls, light_model_cls):
    train_dl, valid_dl = ochuman_keypoints_dls
    model = keypoint_rcnn.model(num_keypoints=19)
    light_model = light_model_cls(model)

    trainer = pl.Trainer(
        max_epochs=1,
        weights_summary=None,
        num_sanity_val_steps=0,
        logger=False,
        checkpoint_callback=False,
    )
    trainer.fit(light_model, train_dl, valid_dl)
