import pytest
from icevision.all import *
from icevision.models.torchvision import retinanet


@pytest.mark.parametrize("metrics", [[COCOMetric()]])
def test_fastai_retinanet_train(
    fridge_faster_rcnn_dls, fridge_retinanet_model, metrics
):
    learn = retinanet.fastai.learner(
        dls=fridge_faster_rcnn_dls, model=fridge_retinanet_model, metrics=metrics
    )

    learn.fine_tune(1, 1e-4)
