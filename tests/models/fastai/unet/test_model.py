import pytest
from icevision.all import *
from icevision.models.fastai.unet.backbones import *


@pytest.mark.parametrize(
    "backbone",
    [resnet18, resnet50, resnet101],
)
def test_unet_model(backbone):
    model = models.fastai.unet.model(
        num_classes=32, img_size=64, backbone=backbone(pretrained=True)
    )

    assert len(list(model.param_groups())) == 6
    assert model[-1][0].out_channels == 32
