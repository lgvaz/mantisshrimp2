"""
This file is redundant in terms of code as it uses the exact same code
as `icevision.models.ultralytics.yolov5.utils.YoloV5BackboneConfig`

We're keeping it to maintain structure, and in case we want to change
something in the future that is multitask model specific
"""

from icevision.models.ultralytics.yolov5.utils import YoloV5BackboneConfig

__all__ = ["YoloV5BackboneConfig"]

# from icevision.imports import *
# from icevision.backbones import BackboneConfig


# class YoloV5MultitaskBackboneConfig(BackboneConfig):
#     def __init__(self, model_name: str):
#         self.model_name = model_name
#         self.pretrained: bool

#     def __call__(self, pretrained: bool = True) -> "YoloV5MultitaskBackboneConfig":
#         """Completes the configuration of the backbone

#         # Arguments
#             pretrained: If True, use a pretrained backbone (on COCO).
#         """
#         self.pretrained = pretrained
#         return self
