import pytest
from icevision.all import *
from icevision.models.torchvision import faster_rcnn


@pytest.mark.parametrize("metrics", [[COCOMetric()]])
def test_fastai_faster_rcnn_train(
    fridge_faster_rcnn_dls, fridge_faster_rcnn_model, metrics
):
    learn = faster_rcnn.fastai.learner(
        dls=fridge_faster_rcnn_dls, model=fridge_faster_rcnn_model, metrics=metrics
    )

    learn.fine_tune(1, 1e-4)
