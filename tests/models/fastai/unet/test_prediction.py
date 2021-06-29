import pytest
from icevision.all import *
from icevision.models.fastai.unet.backbones import *


def _test_preds(preds):
    assert len(preds) == 1
    assert isinstance(preds[0].segmentation.mask_array, MaskArray)
    assert preds[0].segmentation.mask_array.data.squeeze().shape == (64, 64)
    assert preds[0].segmentation.class_map is not None
    assert preds[0].segmentation.class_map.num_classes == 32


@pytest.mark.parametrize(
    "backbone",
    [resnet18, resnet50, resnet101],
)
def test_unet_predict(camvid_ds, backbone):
    _, valid_ds = camvid_ds
    model = models.fastai.unet.model(
        num_classes=32, img_size=64, backbone=backbone(pretrained=True)
    )
    preds = models.fastai.unet.predict(model, valid_ds)
    _test_preds(preds)


@pytest.mark.parametrize(
    "backbone",
    [resnet18, resnet50, resnet101],
)
def test_unet_predict_from_dl(camvid_ds, backbone):
    _, valid_ds = camvid_ds
    infer_dl = models.fastai.unet.infer_dl(valid_ds, batch_size=1, shuffle=False)
    model = models.fastai.unet.model(
        num_classes=32, img_size=64, backbone=backbone(pretrained=True)
    )
    preds = models.fastai.unet.predict_from_dl(model, infer_dl)
    _test_preds(preds)
