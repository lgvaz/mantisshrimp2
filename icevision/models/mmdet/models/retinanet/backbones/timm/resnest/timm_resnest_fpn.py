__all__ = [
    "resnest50d",
]

from icevision.models.mmdet.models.retinanet.backbones.timm.common import *
from icevision.models.mmdet.backbones.timm.resnet import *

from icevision.imports import *
from icevision.models.mmdet.utils import *

base_config_path = mmdet_configs_path / "retinanet"

resnest50d = MMDetTimmRetinanetBackboneConfig(
    config_path=base_config_path / "retinanet_r50_caffe_fpn_1x_coco.py",
    backbone_dict={
        "type": "ResNest50D_Timm",
        "pretrained": True,
        "out_indices": (2, 3, 4),
        "norm_eval": True,
        "frozen_stem": True,
        "frozen_stages": 1,
    },
    weights_url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-resnest/resnest50-528c19ca.pth",
)
