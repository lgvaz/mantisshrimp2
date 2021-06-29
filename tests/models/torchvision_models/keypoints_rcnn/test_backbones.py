import pytest
from icevision.all import *
from icevision.models.torchvision import keypoint_rcnn


@pytest.mark.parametrize(
    "model_name,param_groups_len",
    (
        ("mobilenet", 6),
        ("resnet18", 7),
        ("resnet18_fpn", 8),
        ("resnext50_32x4d_fpn", 8),
        ("wide_resnet50_2_fpn", 8),
    ),
)
def test_keypoint_rcnn_fpn_backbones(model_name, param_groups_len):
    backbone_fn = getattr(models.torchvision.keypoint_rcnn.backbones, model_name)
    backbone = backbone_fn(pretrained=False)

    model = keypoint_rcnn.model(num_keypoints=2, num_classes=4, backbone=backbone)
    assert len(model.param_groups()) == param_groups_len
