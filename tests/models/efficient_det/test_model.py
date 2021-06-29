import pytest
from icevision.all import *
from icevision.models.ross import efficientdet
from icevision.models.ross.efficientdet.backbones import *


@pytest.mark.parametrize(
    "backbone",
    [
        tf_lite0,
        d0,
        d1,
        d2,
        tf_d0_ap,
        tf_d1_ap,
        tf_d2_ap,
        tf_d0,
        tf_d1,
        tf_d2,
    ],
)
def test_efficient_det_param_groups(backbone):
    model = efficientdet.model(
        backbone=backbone(pretrained=False),
        num_classes=42,
        img_size=256,
    )

    assert len(list(model.param_groups())) == 3
