import pytest
from icevision.all import *
from icevision.models.ultralytics.yolov5.backbones import *


@pytest.mark.parametrize(
    "backbone",
    [small, medium, large, extra_large],
)
def test_show_results(fridge_ds, backbone, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    _, valid_ds = fridge_ds
    model = models.ultralytics.yolov5.model(
        num_classes=5, img_size=384, backbone=backbone(pretrained=True)
    )
    models.ultralytics.yolov5.show_results(model, valid_ds)


@pytest.mark.parametrize(
    "backbone",
    [small, medium, large, extra_large],
)
def test_plot_losses(fridge_ds, backbone, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    _, ds = fridge_ds
    model = models.ultralytics.yolov5.model(
        num_classes=5, img_size=384, backbone=backbone(pretrained=True)
    )

    samples_plus_losses, preds, _ = models.ultralytics.yolov5.interp.plot_top_losses(
        model=model, dataset=ds, sort_by="loss_yolo", n_samples=2
    )
    assert len(samples_plus_losses) == len(ds) == len(preds)

    loss_per_image = models.get_samples_losses(samples_plus_losses)
    assert "filepath" in loss_per_image[0].keys()
