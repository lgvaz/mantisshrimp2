import pytest
from icevision.all import *
from icevision.models.torchvision import retinanet


@pytest.fixture(scope="session")
def fake_faster_rcnn_model():
    class FakeRetinanetModel(nn.Module):
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

    return FakeRetinanetModel()


def test_retinanet_show_results(fake_faster_rcnn_model, fridge_ds, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    train_ds, valid_ds = fridge_ds

    retinanet.show_results(
        model=fake_faster_rcnn_model, dataset=valid_ds, num_samples=1, ncols=1
    )
