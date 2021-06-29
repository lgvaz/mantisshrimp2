import pytest
from icevision.all import *
from icevision.models.fastai.unet.backbones import *


@pytest.mark.parametrize("backbone", [resnet18])
def test_fastai_unet_train(camvid_ds, backbone):
    train_ds, valid_ds = camvid_ds
    train_dl = models.fastai.unet.train_dl(
        train_ds, batch_size=4, num_workers=0, shuffle=False
    )
    valid_dl = models.fastai.unet.valid_dl(
        valid_ds, batch_size=1, num_workers=0, shuffle=False
    )
    model = models.fastai.unet.model(
        num_classes=32, img_size=64, backbone=backbone(pretrained=True)
    )

    learn = models.fastai.unet.fastai.learner(dls=[train_dl, valid_dl], model=model)

    learn.fine_tune(1, 1e-4)
