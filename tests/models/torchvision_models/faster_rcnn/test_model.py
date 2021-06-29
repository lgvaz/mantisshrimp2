from icevision.all import *
from icevision.models.torchvision import faster_rcnn
import albumentations as A
import pytest


@pytest.mark.parametrize(
    "model_name,param_groups_len",
    (
        ("resnet18_fpn", 7),
        ("resnext50_32x4d_fpn", 7),
        ("wide_resnet50_2_fpn", 7),
    ),
)
def test_e2e_detect(samples_source, fridge_class_map, model_name, param_groups_len):
    img_path = samples_source / "fridge/odFridgeObjects/images/10.jpg"
    tfms_ = tfms.A.Adapter([A.Resize(384, 384), A.Normalize()])

    backbone_fn = getattr(models.torchvision.faster_rcnn.backbones, model_name)
    backbone = backbone_fn(pretrained=False)
    model = faster_rcnn.model(num_classes=4, backbone=backbone)

    bboxes = faster_rcnn.end2end_detect(
        img_path, tfms_, model, fridge_class_map, detection_threshold=1
    )
    assert len(bboxes) == 0


def test_faster_rcnn_default_param_groups():
    model = faster_rcnn.model(num_classes=4)

    param_groups = model.param_groups()
    assert len(param_groups) == 8


def test_faster_rcnn_mobile_param_groups():
    backbone = models.torchvision.faster_rcnn.backbones.mobilenet(pretrained=False)
    model = faster_rcnn.model(num_classes=6, backbone=backbone)

    param_groups = model.param_groups()
    assert len(param_groups) == 6
