__all__ = [
    "BaseResNet",
    "ResNest50D_Timm",
]

from icevision.models.mmdet.backbones.timm.common import *
from mmdet.models.builder import BACKBONES

from typing import Optional, Collection
from torch.nn.modules.batchnorm import _BatchNorm
from typing import Tuple, Collection, List

import timm
from timm.models.resnest import *
from timm.models.registry import *

import torch.nn as nn
import torch


class BaseResNet(MMDetTimmBase):
    """
    Base class that implements model freezing and forward methods
    """

    def __init__(
        self,
        model_name: str = None,
        pretrained: bool = True,  # doesn't matter
        out_indices: Collection[int] = (2, 3, 4),
        norm_eval: bool = True,
        frozen_stages: int = 1,
        frozen_stem: bool = True,
    ):

        super().__init__(
            model_name=model_name,
            pretrained=pretrained,
            out_indices=out_indices,
            norm_eval=norm_eval,
        )
        self.frozen_stages = frozen_stages
        self.frozen_stem = frozen_stem

    def test_pretrained_weights(self):
        # Get model method from the timm registry by model name
        model_fn = model_entrypoint(self.model_name)
        model = model_fn(pretrained=self.pretrained)
        assert torch.equal(self.model.conv1.weight, model.conv1.weight)

    def post_init_setup(self):
        self.freeze(
            freeze_stem=self.frozen_stem,
            freeze_blocks=self.frozen_stages,
        )
        # self.test_pretrained_weights()

    def freeze(self, freeze_stem: bool = True, freeze_blocks: int = 1):
        "Optionally freeze the stem and/or Inverted Residual blocks of the model"
        if 0 > freeze_blocks > 4:
            raise ValueError("freeze_blocks values must between 0 and 3 included")

        m = self.model

        # Stem freezing logic
        if freeze_stem:
            for l in [m.conv1, m.bn1]:
                l.eval()
                for param in l.parameters():
                    param.requires_grad = False

        # `freeze_blocks=1` freezes the first block, and so on
        for i in range(1, self.frozen_stages + 1):
            l = getattr(m, f"layer{i}")
            l.eval()
            for param in l.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        "Convert the model to training mode while optionally freezing BatchNorm"
        super(BaseResNet, self).train(mode)
        self.freeze(
            freeze_stem=self.frozen_stem,
            freeze_blocks=self.frozen_stages,
        )
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()


@BACKBONES.register_module(force=True)
class ResNest50D_Timm(BaseResNet):
    def __init__(
        self,
        pretrained: bool = True,  # doesn't matter
        out_indices: Collection[int] = (2, 3, 4),
        norm_eval: bool = True,
        frozen_stages: int = 1,
        frozen_stem: bool = True,
    ):
        "ResNe(s)t50 with hardcoded `pretrained=True`"
        super().__init__(
            model_name="resnest50d",
            pretrained=pretrained,
            out_indices=out_indices,
            norm_eval=norm_eval,
            frozen_stages=frozen_stages,
            frozen_stem=frozen_stem,
        )

        self.weights_url = "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-resnest/resnest50-528c19ca.pth"

        self.post_init_setup()


@BACKBONES.register_module(force=True)
class ResNest101E_Timm(BaseResNet):
    def __init__(
        self,
        pretrained: bool = True,  # doesn't matter
        out_indices: Collection[int] = (2, 3, 4),
        norm_eval: bool = True,
        frozen_stages: int = 1,
        frozen_stem: bool = True,
    ):
        "ResNe(s)t101 with hardcoded `pretrained=True`"
        super().__init__(
            model_name="resnest101e",
            pretrained=pretrained,
            out_indices=out_indices,
            norm_eval=norm_eval,
            frozen_stages=frozen_stages,
            frozen_stem=frozen_stem,
        )

        self.weights_url = "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-resnest/resnest101-22405ba7.pth"

        self.post_init_setup()
