import torch
import torch.nn as nn
import numpy as np

from torch import Tensor
from torch.nn import Parameter

from typing import Collection, Union, List, Tuple
from icevision.utils.torch_utils import params
from icevision.utils.utils import flatten
from loguru import logger

logger = logger.opt(colors=True)

__all__ = ["FreezingInterfaceExtension"]


class FreezingInterfaceExtension:
    """
    Model freezing and unfreezing extensions for `HybridYOLOV5`
    Note that the BatchNorm layers are also frozen, but that part is not
    defined here, but in the main module's `.train()` method directly
    """

    def _get_params_stem(self) -> List[nn.Parameter]:
        return params(self.model[0])

    def _get_params_backbone(self) -> List[List[Parameter]]:
        return [params(m) for m in self.model[1 : self.bbone_blocks_end_idx]]

    def _get_params_neck(self) -> List[List[Parameter]]:
        return [params(m) for m in self.model[self.bbone_blocks_end_idx :][:-1]]

    def _get_params_bbox_head(self) -> List[List[Parameter]]:
        return params(self.model[-1])

    def _get_params_classifier_heads(self) -> List[List[Parameter]]:
        return [params(self.classifier_heads)]

    def _set_param_grad_stem(self, mode: bool):
        for p in flatten(self._get_params_stem()):
            p.requires_grad = mode

    def _set_param_grad_backbone(self, mode: bool, bbone_blocks: Collection[int]):
        error_msg = f"""
        `bbone_blocks` must be a list|tuple of values between 0-{self.num_bbone_blocks} specifying which blocks to set this state for
        """

        if not isinstance(bbone_blocks, (list, tuple)):
            raise TypeError(error_msg)
        if not all(isinstance(x, int) for x in bbone_blocks):
            raise TypeError(error_msg)
        if not bbone_blocks == []:
            if not 0 <= bbone_blocks[0] <= self.num_bbone_blocks - 1:
                raise ValueError(error_msg)

        pgs = np.array(self._get_params_backbone(), dtype="object")
        for p in flatten(pgs[bbone_blocks]):
            p.requires_grad = mode

    def _set_param_grad_neck(self, mode: bool):
        for p in flatten(self._get_params_neck()):
            p.requires_grad = mode

    def _set_param_grad_bbox_head(self, mode: bool):
        for p in flatten(self._get_params_bbox_head()):
            p.requires_grad = mode

    def _set_param_grad_classifier_heads(self, mode: bool):
        for p in flatten(self._get_params_classifier_heads()):
            p.requires_grad = mode

    def freeze(
        self,
        stem: bool = False,
        bbone_blocks: int = 0,  # between 0 to self.num_bbone_blocks
        neck: bool = False,
        bbox_head: bool = False,
        classifier_heads: bool = False,
    ):
        """
        Freeze selected parts of the network. By default, none of the parts are frozen, you need
        to manually set each arg's value to `True` if you want to freeze it. If you don't want
        this fine grained control, see `.freeze_detector()`, `.freeze_backbone()`, `.freeze_classifier_heads()`

        Args:
            stem (bool, optional): Freeze the first conv layer. Defaults to True.
            bbone_blocks (int, optional): Number of blocks to freeze. If 0, none are frozen; if ==self.num_bbone_blocks, all are frozen.
            neck (bool, optional): Freeze the neck (FPN). Defaults to False.
            bbox_head (bool, optional): Freeze the bounding box head (the `Detect` module). Defaults to False.
            classifier_heads (bool, optional): Freeze all the classification heads. Defaults to False.
        """
        if stem:
            self._set_param_grad_stem(False)
        if bbone_blocks:
            self._set_param_grad_backbone(False, [i for i in range(bbone_blocks)])
        if neck:
            self._set_param_grad_neck(False)
        if bbox_head:
            self._set_param_grad_bbox_head(False)
        if classifier_heads:
            self._set_param_grad_classifier_heads(False)

    def unfreeze(
        self,
        stem: bool = False,
        bbone_blocks: int = 0,
        neck: bool = False,
        bbox_head: bool = False,
        classifier_heads: bool = False,
    ):
        """
        Unfreeze specific parts of the model. By default all parts are kept frozen.
        You need to manually set whichever part you want to unfreeze by passing that arg as `True`.
        See `.unfreeze_detector()`, `.unfreeze_backbone()`, `.unfreeze_classifier_heads()` methods if you
        don't want this fine grained control.

        Note that `bbone_blocks` works differently from `.freeze()`. `bbone_blocks=3` will unfreeze
        the _last 3_ blocks, and `bbone_blocks=self.num_bbone_blocks` will unfreeze _all_ the blocks
        """
        if stem:
            self._set_param_grad_stem(True)
        if bbone_blocks:
            self._set_param_grad_backbone(
                True,
                [
                    i
                    for i in range(
                        self.num_bbone_blocks - bbone_blocks, self.num_bbone_blocks
                    )
                ],
            )
        if neck:
            self._set_param_grad_neck(True)
        if bbox_head:
            self._set_param_grad_bbox_head(True)
        if classifier_heads:
            self._set_param_grad_classifier_heads(True)

    def freeze_detector(self):
        "Freezes the entire detector i.e. stem, bbone, neck, bbox head"
        self.freeze(
            stem=True, bbone_blocks=self.num_bbone_blocks, neck=True, bbox_head=True
        )

    def unfreeze_detector(self):
        "Unfreezes the entire detector i.e. stem, bbone, neck, bbox head"
        self.unfreeze(
            stem=True, bbone_blocks=self.num_bbone_blocks, neck=True, bbox_head=True
        )

    def freeze_backbone(self, fpn=True):
        "Freezes the entire backbone, optionally without the neck/fpn"
        self.freeze(
            stem=True, bbone_blocks=self.num_bbone_blocks, neck=True if fpn else False
        )

    def unfreeze_backbone(self, fpn=True):
        "Unfreezes the entire backbone, optionally without the neck/fpn"
        self.unfreeze(
            stem=True, bbone_blocks=self.num_bbone_blocks, neck=True if fpn else False
        )

    def freeze_classifier_heads(self):
        "Freezes just the classification heads"
        self.freeze(classifier_heads=True)

    def unfreeze_classifier_heads(self):
        "Unfreezes just the classification heads"
        self.unfreeze(classifier_heads=True)

    def freeze_specific_classifier_heads(
        self, names: Union[str, List[str], None] = None, _grad: bool = False
    ):
        "Freeze all, one or a few classifier heads"
        if isinstance(names, str):
            names = [names]
        if names is None:
            names = list(self.classifier_heads.keys())

        for name in names:
            for p in flatten(params(self.classifier_heads[name])):
                p.requires_grad = _grad

    def unfreeze_specific_classifier_heads(
        self, names: Union[str, List[str], None] = None
    ):
        self.freeze_specific_classifier_heads(names=names, _grad=True)
