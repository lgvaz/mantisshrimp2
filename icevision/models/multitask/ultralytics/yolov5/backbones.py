"""
This file is redundant in terms of code as it uses the exact same code
as `icevision.models.ultralytics.yolov5.backbones`

We're keeping it to maintain structure, and in case we want to change
something in the future that is multitask model specific
"""

from icevision.models.multitask.ultralytics.yolov5.utils import *
from icevision.models.ultralytics.yolov5.backbones import *

__all__ = [
    "small",
    "medium",
    "large",
    "extra_large",
]
