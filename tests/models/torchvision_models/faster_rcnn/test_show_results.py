import pytest
from icevision.all import *
from icevision.models.torchvision import faster_rcnn


@pytest.fixture(scope="session")
def fake_faster_rcnn_model():
    class FakeFasterRCNNModel(nn.Module):
        def __init__(self):
            super().__init__()
            # hack for the function `model_device` to work
            self.layer = nn.Linear(1, 1)

        def forward(self, *args, **kwargs):
            return [
                {
                    "scores": tensor([0.8, 0.9]),
                    "labels": tensor([1, 2]),
                    "boxes": tensor([[10, 10, 50, 50], [20, 20, 40, 40]]),
                }
            ]

    return FakeFasterRCNNModel()


def test_faster_rcnn_show_results(fake_faster_rcnn_model, fridge_ds, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    _, valid_ds = fridge_ds

    faster_rcnn.show_results(
        model=fake_faster_rcnn_model, dataset=valid_ds, num_samples=1, ncols=1
    )


def test_plot_losses(fridge_faster_rcnn_model, fridge_ds, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    model = fridge_faster_rcnn_model
    ds, _ = fridge_ds
    by = {
        "method": "weighted",
        "weights": {
            "loss_box_reg": 1,
            "loss_classifier": 0,
            "loss_objectness": 0,
            "loss_rpn_box_reg": 0,
        },
    }

    samples_plus_losses, preds, _ = faster_rcnn.interp.plot_top_losses(
        model=model, dataset=ds, sort_by=by, n_samples=2
    )
    assert len(samples_plus_losses) == len(ds) == len(preds)


def test_get_losses(fridge_faster_rcnn_model, fridge_ds):
    model = fridge_faster_rcnn_model
    ds, _ = fridge_ds

    samples, losses_stats = faster_rcnn.interp.get_losses(model, ds)
    assert len(samples) == len(ds)
    assert set(losses_stats.keys()) == {
        "loss_box_reg",
        "loss_classifier",
        "loss_objectness",
        "loss_rpn_box_reg",
        "loss_total",
    }
    assert "loss_box_reg" in samples[0].losses.keys()
    assert "text" not in samples[0].losses.keys()


def test_add_annotations(fridge_faster_rcnn_model, fridge_ds):
    model = fridge_faster_rcnn_model
    ds, _ = fridge_ds

    samples, _ = faster_rcnn.interp.get_losses(model, ds)
    samples = models.add_annotations(samples)
    assert "loss_classifier" in samples[0].losses["text"]
    assert "IMG" in samples[0].losses["text"]


def test_get_samples_losses(fridge_faster_rcnn_model, fridge_ds, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    model = fridge_faster_rcnn_model
    ds, _ = fridge_ds

    samples_plus_losses, _, _ = faster_rcnn.interp.plot_top_losses(
        model=model, dataset=ds, n_samples=2
    )
    loss_per_image = models.get_samples_losses(samples_plus_losses)
    assert "filepath" in loss_per_image[0].keys()
