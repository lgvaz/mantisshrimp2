from icevision.all import *
from icevision.models.torchvision import keypoint_rcnn


def test_keypoints_rcnn_model():
    model = keypoint_rcnn.model(num_keypoints=1)
    param_groups = model.param_groups()

    assert len(param_groups) == 8
    assert model.roi_heads.keypoint_predictor.kps_score_lowres.out_channels == 1

    backbone = models.torchvision.keypoint_rcnn.backbones.resnet18_fpn(pretrained=True)
    model = keypoint_rcnn.model(backbone=backbone, num_keypoints=10)

    assert len(param_groups) == 8
    assert model.roi_heads.keypoint_predictor.kps_score_lowres.out_channels == 10
