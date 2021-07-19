__all__ = [
    "resnet50",
    "resnetrs50",
]

from icevision.models.mmdet.models.retinanet.backbones.timm.common import *
from icevision.models.mmdet.backbones.timm.resnet import *

from icevision.imports import *
from icevision.models.mmdet.utils import *
from icevision.models.mmdet.utils import *

base_config_path = mmdet_configs_path / "retinanet"

resnet50 = MMDetTimmRetinanetBackboneConfig(
    config_path=base_config_path / "retinanet_r50_caffe_fpn_1x_coco.py",
    backbone_dict={
        "type": "ResNet50_Timm",
        "pretrained": True,
        "out_indices": (2, 3, 4),
        "norm_eval": True,
        "frozen_stem": True,
        "frozen_stages": 1,
    },
    weights_url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet50_ram-a26f946b.pth",
)

resnetrs50 = MMDetTimmRetinanetBackboneConfig(
    config_path=base_config_path / "retinanet_r50_caffe_fpn_1x_coco.py",
    backbone_dict={
        "type": "ResNetRS50_Timm",
        "pretrained": True,
        "out_indices": (2, 3, 4),
        "norm_eval": True,
        "frozen_stem": True,
        "frozen_stages": 1,
    },
    weights_url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rs-weights/resnetrs50_ema-6b53758b.pth",
)
