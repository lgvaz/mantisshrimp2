__all__ = [
    # "mobilenetv2_100",
    # "mobilenetv2_110d",
    # "mobilenetv2_120d",
    # "mobilenetv2_140",
    # "mobilenetv3_large_075",
    "mobilenetv3_large_100",
    "mobilenetv3_rw",
    # "mobilenetv3_small_075",
    # "mobilenetv3_small_100",
    # "tf_mobilenetv3_large_075",
    # "tf_mobilenetv3_large_100",
    # "tf_mobilenetv3_large_minimal_100",
    # "tf_mobilenetv3_small_075",
    # "tf_mobilenetv3_small_100",
    # "tf_mobilenetv3_small_minimal_100",
]

from icevision.models.mmdet.models.retinanet.backbones.timm.common import *
from icevision.models.mmdet.backbones.timm.mobilenet import *

from icevision.imports import *
from icevision.models.mmdet.utils import *

base_config_path = mmdet_configs_path / "retinanet"

mobilenetv3_large_100 = MMDetTimmRetinanetBackboneConfig(
    config_path=base_config_path / "retinanet_r50_caffe_fpn_1x_coco.py",
    backbone_dict={
        "type": "MobileNetV3_Large_100",
        "pretrained": True,
        "out_indices": (2, 3, 4),
        "norm_eval": True,
        "frozen_stem": True,
        "frozen_stages": 1,
    },
    weights_url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mobilenetv3_large_100_ra-f55367f5.pth",
)

mobilenetv3_rw = MMDetTimmRetinanetBackboneConfig(
    config_path=base_config_path / "retinanet_r50_caffe_fpn_1x_coco.py",
    backbone_dict={
        "type": "MobileNetV3_RW",
        "pretrained": True,
        "out_indices": (2, 3, 4),
        "norm_eval": True,
        "frozen_stem": True,
        "frozen_stages": 1,
    },
    weights_url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mobilenetv3_100-35495452.pth",
)
