import pytest
from icevision.all import *
from icevision.models.torchvision import keypoint_rcnn


def test_fastai_keypoints_rcnn_train(ochuman_keypoints_dls):
    model = keypoint_rcnn.model(num_keypoints=19)
    learn = keypoint_rcnn.fastai.learner(dls=[*ochuman_keypoints_dls], model=model)

    learn.fine_tune(1, 1e-4)
