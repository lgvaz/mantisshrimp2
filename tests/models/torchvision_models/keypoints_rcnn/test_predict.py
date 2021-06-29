import pytest
from icevision.all import *
from icevision.models.torchvision import keypoint_rcnn


def test_predict_keypoints_rcnn_train(ochuman_ds):
    _, valid_ds = ochuman_ds
    model = keypoint_rcnn.model(num_keypoints=19)

    infer_dl = keypoint_rcnn.infer_dl(valid_ds, batch_size=2)
    preds = keypoint_rcnn.predict_from_dl(model=model, infer_dl=infer_dl)
    p = preds[0].pred

    assert len(infer_dl) == 1
    assert len(preds) == 2
    assert len(list(p.detection.keypoints_scores)) == len(p.detection.keypoints)
    assert len(p.detection.bboxes) == len(p.detection.keypoints)
