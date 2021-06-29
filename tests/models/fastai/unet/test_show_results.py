import pytest
from icevision.all import *
from icevision.models.fastai.unet.backbones import *
from icevision.models.interpretation import get_samples_losses


@pytest.mark.parametrize(
    "backbone",
    [resnet18, resnet50, resnet101],
)
def test_show_results(camvid_ds, backbone, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    _, valid_ds = camvid_ds
    model = models.fastai.unet.model(
        num_classes=32, img_size=64, backbone=backbone(pretrained=True)
    )
    models.fastai.unet.show_results(model, valid_ds)


@pytest.mark.parametrize(
    "backbone",
    [resnet18, resnet50, resnet101],
)
def test_plot_losses(camvid_ds, backbone, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    _, ds = camvid_ds
    model = models.fastai.unet.model(
        num_classes=32, img_size=64, backbone=backbone(pretrained=True)
    )

    samples_plus_losses, preds, _ = models.fastai.unet.interp.plot_top_losses(
        model=model, dataset=ds, sort_by="loss_unet", n_samples=2
    )
    assert len(samples_plus_losses) == len(ds) == len(preds)

    loss_per_image = get_samples_losses(samples_plus_losses)
    assert "filepath" in loss_per_image[0].keys()
