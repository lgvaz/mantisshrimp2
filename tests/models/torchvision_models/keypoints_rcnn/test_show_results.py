import pytest
from icevision.all import *
from icevision.models.torchvision import keypoint_rcnn


def test_keypoints_rcnn_show_results(ochuman_ds, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    train_ds, valid_ds = ochuman_ds
    model = keypoint_rcnn.model(num_keypoints=19)

    keypoint_rcnn.show_results(model=model, dataset=valid_ds, num_samples=1, ncols=1)
