# Hacked together by Rahul & Farid

from mmcv.cnn import MODELS as MMCV_MODELS
from mmcv.utils import Registry

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Union, Optional, Dict
from torch import Tensor
from functools import partial
from collections import namedtuple
from dataclasses import dataclass

TensorList = List[Tensor]
TensorDict = Dict[str, Tensor]

MODELS = Registry("models", parent=MMCV_MODELS)
CLASSIFICATION_HEADS = MODELS

__all__ = ["ImageClassificationHead", "ClassifierConfig"]


class Passthrough(nn.Module):
    def forward(self, x):
        return x


"""
`ClassifierConfig` is useful to instantiate `ImageClassificationHead`
in different settings. If using `mmdet`, we don't use this as the config
is then a regular dictionary.

When using yolov5, we can easily pass around this config to create the model
Often, it'll be used inside a dictionary of configs
"""


@dataclass
class ClassifierConfig:
    # classifier_name: str
    out_classes: int
    num_fpn_features: int = 512
    fpn_keys: Union[List[str], List[int], None] = None
    dropout: Optional[float] = 0.2
    pool_inputs: bool = True
    # Loss function args
    loss_func: Optional[nn.Module] = None
    activation: Optional[nn.Module] = None
    multilabel: bool = False
    loss_func_wts: Optional[Tensor] = None
    loss_weight: float = 1.0
    # Post activation processing
    thresh: Optional[float] = None
    topk: Optional[int] = None

    def __post_init__(self):
        if isinstance(self.fpn_keys, int):
            self.fpn_keys = [self.fpn_keys]

        if self.loss_func_wts is not None:
            if not self.multilabel:
                self.loss_func_wts = self.loss_func_wts.to(torch.float32)
            if torch.cuda.is_available():
                self.loss_func_wts = self.loss_func_wts.cuda()

        if self.multilabel:
            if self.topk is None and self.thresh is None:
                self.thresh = 0.5
        else:
            if self.topk is None and self.thresh is None:
                self.topk = 1


@CLASSIFICATION_HEADS.register_module(name="ImageClassificationHead")
class ImageClassificationHead(nn.Module):
    """
    Image classification head that optionally takes `fpn_keys` features from
    an FPN, average pools and concatenates them into a single tensor
    of shape `num_features` and then runs a linear layer to `out_classes

    fpn_features: [List[Tensor]] => AvgPool => Flatten => Linear`

    Also includes `compute_loss` to match the design of other
    components of object detection systems.
    To use your own loss function, pass it into `loss_func`.
    If `loss_func` is None (by default), we create one based on other args:
    If `multilabel` is true, one-hot encoded targets are expected and
    nn.BCEWithLogitsLoss is used, else nn.CrossEntropyLoss is used
    and targets are expected to be integers
    NOTE: Not all loss function args are exposed
    """

    def __init__(
        self,
        out_classes: int,
        num_fpn_features: int,
        fpn_keys: Union[List[str], List[int], None] = None,
        dropout: Optional[float] = 0.2,
        pool_inputs: bool = True,  # ONLY for advanced use cases where input feature maps are already pooled
        # Loss function args
        loss_func: Optional[nn.Module] = None,
        activation: Optional[nn.Module] = None,
        multilabel: bool = False,
        loss_func_wts: Optional[Tensor] = None,
        loss_weight: float = 1.0,
        # Final postprocessing args
        thresh: Optional[float] = None,
        topk: Optional[int] = None,
    ):
        super().__init__()

        # Setup loss function & activation
        self.multilabel = multilabel
        self.loss_func, self.loss_func_wts, self.loss_weight = (
            loss_func,
            loss_func_wts,
            loss_weight,
        )
        self.activation = activation
        self.pool_inputs = pool_inputs
        self.thresh, self.topk = thresh, topk

        # Setup head
        self.fpn_keys = fpn_keys

        layers = [
            nn.Dropout(dropout) if dropout else Passthrough(),
            nn.Linear(num_fpn_features, out_classes),
        ]
        layers.insert(0, nn.Flatten(1)) if self.pool_inputs else None
        self.classifier = nn.Sequential(*layers)

        self.setup_loss_function()
        self.setup_postprocessing()

    def setup_postprocessing(self):
        if self.multilabel:
            if self.topk is None and self.thresh is None:
                self.thresh = 0.5
        else:
            if self.topk is None and self.thresh is None:
                self.topk = 1

    def setup_loss_function(self):
        if self.loss_func is None:
            if self.multilabel:
                self.loss_func = nn.BCEWithLogitsLoss(pos_weight=self.loss_func_wts)
                # self.loss_func = partial(
                #     F.binary_cross_entropy_with_logits, pos_weight=self.loss_func_wts
                # )
                self.activation = nn.Sigmoid()
                # self.activation = torch.sigmoid  # nn.Sigmoid()
            else:
                # self.loss_func = nn.CrossEntropyLoss(self.loss_func_wts)
                self.loss_func = nn.CrossEntropyLoss(weight=self.loss_func_wts)
                # self.loss_func = partial(F.cross_entropy, weight=self.loss_func_wts)
                self.activation = nn.Softmax(-1)
                # self.activation = partial(F.softmax, dim=-1)  # nn.Softmax(-1)

    @classmethod
    def from_config(cls, config: ClassifierConfig):
        return cls(**config.__dict__)

    # TODO: Make it run with regular features as well
    def forward(self, features: Union[Tensor, TensorDict, TensorList]):
        """
        Sequence of outputs from an FPN or regular feature extractor
        => Avg. Pool each into 1 dimension
        => Concatenate into single tensor
        => Linear layer -> output classes

        If `self.fpn_keys` is specified, it grabs the specific (int|str) indices from
        `features` for the pooling layer, else it takes _all_ of them
        """
        if isinstance(features, (list, dict, tuple)):
            # Grab specific features if specified
            if self.fpn_keys is not None:
                pooled_features = [
                    F.adaptive_avg_pool2d(features[k], 1) for k in self.fpn_keys
                ]
            # If no `fpn_keys` exist, concat all the feature maps (could be expensive)
            else:
                pooled_features = [F.adaptive_avg_pool2d(feat, 1) for feat in features]
            pooled_features = torch.cat(pooled_features, dim=1)

        # If doing regular (non-FPN) feature extraction, we don't need `fpn_keys` and
        # just avg. pool the last layer's features
        elif isinstance(features, Tensor):
            pooled_features = (
                F.adaptive_avg_pool2d(features, 1) if self.pool_inputs else features
            )
        else:
            raise TypeError(
                f"Expected TensorList|TensorDict|Tensor|tuple, got {type(features)}"
            )

        return self.classifier(pooled_features)

    # TorchVision style API
    def compute_loss(self, predictions, targets):
        return self.loss_weight * self.loss_func(predictions, targets)

    def postprocess(self, predictions):
        return self.activation(predictions)

    # MMDet style API
    def forward_train(self, x, gt_label) -> Tensor:
        preds = self(x)
        return self.loss_weight * self.loss_func(preds, gt_label)

    def forward_activate(self, x):
        "Run forward pass with activation function"
        x = self(x)
        return self.activation(x)
