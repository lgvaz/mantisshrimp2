"""
The following imports are not strictly necessary if you're only using the high level API,
  but are nice to have for quick dev / debugging, and to use some of the lower level APIs.
  They must be imported before the rest of the imports to respect namespaces
"""

import icevision.tfms as tfms
from icevision.models.multitask.classification_heads import *
from icevision.models.multitask.utils import *
from icevision.data.dataset import Dataset
from yolov5.utils.loss import ComputeLoss


"""
The following imports are what are essential for the high level API, and in line with
  the way modules are imported with the rest of the library
"""

from icevision.models.multitask.ultralytics.yolov5.dataloaders import *
from icevision.models.multitask.ultralytics.yolov5.model import *
from icevision.models.multitask.ultralytics.yolov5.prediction import *
from icevision.models.multitask.ultralytics.yolov5.utils import *
from icevision.models.multitask.ultralytics.yolov5.backbones import *
from icevision.models.multitask.ultralytics.yolov5.arch.yolo_hybrid import *


from icevision.soft_dependencies import SoftDependencies

if SoftDependencies.fastai:
    from icevision.models.multitask.ultralytics.yolov5 import fastai

if SoftDependencies.pytorch_lightning:
    from icevision.models.multitask.ultralytics.yolov5 import lightning
