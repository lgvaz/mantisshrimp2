from typing import Dict, List
from collections import OrderedDict
from icevision.models.multitask.classification_heads import *


import torch
import torch.nn as nn
import torch.distributed as dist
from torch import Tensor

from icevision.models.mmdet.utils import *
from mmcv import Config, ConfigDict
from mmdet.models.builder import DETECTORS
from mmdet.models.builder import build_backbone, build_detector, build_head, build_neck
from mmdet.models.detectors.single_stage import SingleStageDetector
from mmdet.core.bbox import *
from typing import Union, List, Dict, Tuple, Optional

from icevision.models.multitask.mmdet.dataloaders import (
    TensorDict,
    ClassificationGroupDataDict,
    DataDictClassification,
    DataDictDetection,
)
import numpy as np
from icevision.models.multitask.utils.model import *
from icevision.models.multitask.utils.dtypes import *


__all__ = [
    "HybridSingleStageDetector",
    "build_backbone",
    "build_detector",
    "build_head",
    "build_neck",
]


@DETECTORS.register_module(name="HybridSingleStageDetector")
class HybridSingleStageDetector(SingleStageDetector):
    # TODO: Add weights for loss functions
    def __init__(
        self,
        backbone: Union[dict, ConfigDict],
        neck: Union[dict, ConfigDict],
        bbox_head: Union[dict, ConfigDict],
        classification_heads: Union[None, dict, ConfigDict] = None,
        # keypoint_heads=None,  # TODO Someday SOON.
        train_cfg: Union[None, dict, ConfigDict] = None,
        test_cfg: Union[None, dict, ConfigDict] = None,
        pretrained=None,
        init_cfg: Union[None, dict, ConfigDict] = None,
    ):
        super(HybridSingleStageDetector, self).__init__(
            # Use `init_cfg` post mmdet 2.12
            # backbone, neck, bbox_head, train_cfg, test_cfg, pretrained, init_cfg
            backbone=ConfigDict(backbone),
            neck=ConfigDict(neck),
            bbox_head=ConfigDict(bbox_head),
            train_cfg=ConfigDict(train_cfg),
            test_cfg=ConfigDict(test_cfg),
            pretrained=pretrained,
            init_cfg=ConfigDict(init_cfg),
        )
        if classification_heads is not None:
            self.classifier_heads = build_classifier_heads(classification_heads)

    def train_step(
        self,
        data: dict,
        step_type: ForwardType = ForwardType.TRAIN,
    ) -> Dict[str, Union[Tensor, TensorDict, int]]:
        """
        A single iteration step (over a batch)
        Args:
            data: The output of dataloader. Typically `self.fwd_train_data_keys` or
                  `self.fwd_eval_data_keys`
            step_type (Enum): ForwardType.TRAIN | ForwardType.EVAL | ForwardType.TRAIN_MULTI_AUG

        Returns:
            dict[str, Union[Tensor, TensorDict, int]]
                * `loss` <Tensor> : summed losses for backprop
                * `log_vars` <TensorDict> : variables to be logged
                * `num_samples` <int> : batch size per GPU when using DDP
        """
        losses = self(data=data, step_type=step_type)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(data["img_metas"])
            if "img_metas" in data.keys()
            else len(data["detection"]["img_metas"]),
        )
        return outputs

    # @auto_fp16(apply_to=("img",))
    def forward(self, data: dict, step_type: ForwardType):
        """
        Calls either `self.forward_train`, `self.forward_eval` or
        `self.forward_multi_aug_train` depending on the value of `step_type`

        No TTA supported unlike all other mmdet models
        """
        if step_type is ForwardType.TRAIN_MULTI_AUG:
            return self.forward_multi_aug_train(data)

        elif step_type is ForwardType.TRAIN:
            return self.forward_train(data, gt_bboxes_ignore=None)

        elif step_type is ForwardType.EVAL:
            return self.forward_eval(data, rescale=False)

        else:
            raise RuntimeError(
                f"Invalid `step_type`. Received: {type(step_type.__class__)}; Expected: {ForwardType.__class__}"
            )

    fwd_multi_aug_train_data_keys = ["detection", "classification"]
    fwd_train_data_keys = [
        "img",
        "gt_bboxes",
        "gt_bbox_labels",
        "gt_classification_labels",
    ]
    fwd_eval_data_keys = ["img", "img_metas"]

    def forward_multi_aug_train(
        self,
        data: Dict[str, Union[DataDictClassification, DataDictDetection]],
    ) -> Dict[str, Tensor]:
        """
        Forward method where multiple views of the same image are passed.
        The model does a dedicated forward pass for the `detection` images
          and dedicated forward passes for each `classification` group. See
          the dataloader docs for more details
        Args:
            data <Dict[str, TensorDict]> : a dictionary with two keys -
              `detection` and `classification`. See the dataloader docs for
              more details on the exact structure

        Returns:
            dict[str, Tensor]
                * `loss_classification`: Dictionary of classification losses where each key
                                         corresponds to the classification head / task name
                * `loss_cls`: Bbox classification loss
                * `loss_bbox`: Bbox regression loss
        """
        assert set(data.keys()).issuperset(self.fwd_multi_aug_train_data_keys)
        # detection_img, img_metas, gt_bboxes, gt_bbox_labels = data["detection"].values()
        super(SingleStageDetector, self).forward_train(
            data["detection"]["img"],
            data["detection"]["img_metas"],
        )
        detection_features = self.extract_feat(data["detection"]["img"])

        losses = self.bbox_head.forward_train(
            x=detection_features,
            img_metas=data["detection"]["img_metas"],
            gt_bboxes=data["detection"]["gt_bboxes"],
            gt_labels=data["detection"]["gt_bbox_labels"],
            # NOTE we do not return `gt_bboxes_ignore` in the dataloader
            gt_bboxes_ignore=data["detection"].get("gt_bboxes_ignore", None),
        )

        # Compute features per _group_, then do a forward pass through each
        # classification head in that group to compute the loss
        classification_losses = {}
        for group, data in data["classification"].items():
            classification_features = self.extract_feat(data["images"])
            for task in data["tasks"]:
                head = self.classifier_heads[task]
                classification_losses[task] = head.forward_train(
                    x=classification_features,
                    gt_label=data["classification_labels"][task],
                )

        losses["loss_classification"] = classification_losses
        return losses

    def forward_train(self, data: dict, gt_bboxes_ignore=None) -> Dict[str, Tensor]:
        """
        Forward pass
        Args:
            img: Normalised input images of shape (N, C, H, W).
            img_metas: A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes: List of gt bboxes in `xyxy` format for each image
            gt_labels: Integer class indices corresponding to each box
            gt_classification_labels: Dict of ground truths per classification task
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]
                * `loss_classification`: Dictionary of classification losses where each key
                                         corresponds to the classification head / task name
                * `loss_cls`: Bbox classification loss
                * `loss_bbox`: Bbox regression loss
        """
        assert set(data.keys()).issuperset(self.fwd_train_data_keys)
        super(SingleStageDetector, self).forward_train(data["img"], data["img_metas"])
        features = self.extract_feat(data["img"])
        losses = self.bbox_head.forward_train(
            x=features,
            img_metas=data["img_metas"],
            gt_bboxes=data["gt_bboxes"],
            gt_labels=data["gt_bbox_labels"],
            gt_bboxes_ignore=gt_bboxes_ignore,
        )

        classification_losses = {
            name: head.forward_train(
                x=features,
                gt_label=data["gt_classification_labels"][name],
            )
            for name, head in self.classifier_heads.items()
        }
        losses["loss_classification"] = classification_losses
        return losses

    # Maintain API
    # Placeholder in case we want to do TTA during eval?
    def simple_test(self, *args):
        return self.forward_eval(*args)

    def forward_eval(
        self, data: dict, rescale: bool = False
    ) -> Dict[str, Union[TensorDict, List[np.ndarray]]]:
        """
        TODO Update mmdet docstring

        Eval / test function on a single image (without TTA). Returns raw predictions of
        the model that can be processed in `convert_raw_predictions`

        Args:
            imgs: List of multiple images
            img_metas: List of image metadata.
            rescale: Whether to rescale the results.

        Returns:
            {
                "bbox_results": List[ArrayList],
                "classification_results": TensorDict
            }

            bbox_results: Nested list of BBox results The outer list corresponds
                          to each image. The inner list
                          corresponds to each class.
            classification_results: Dictionary of activated outputs for each classification head
        """
        assert set(data.keys()).issuperset(self.fwd_eval_data_keys)
        # Raw outputs from network
        img, img_metas = data["img"], data["img_metas"]
        features = self.extract_feat(img)
        bbox_outs = self.bbox_head(features)
        classification_results = {
            name: head.forward_activate(features)
            for name, head in self.classifier_heads.items()
        }

        # Get original input shape to support onnx dynamic shape
        if torch.onnx.is_in_onnx_export():
            # get shape as tensor
            img_shape = torch._shape_as_tensor(img)[2:]
            img_metas[0]["img_shape_for_onnx"] = img_shape

        bbox_list = self.bbox_head.get_bboxes(*bbox_outs, img_metas, rescale=rescale)

        # Skip post-processing when exporting to ONNX
        if torch.onnx.is_in_onnx_export():
            return bbox_list, classification_results

        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return {
            "bbox_results": bbox_results,
            "classification_results": classification_results,
        }

    # NOTE: This is experimental
    def forward_onnx(self, one_img: Tensor, one_img_metas: List[ImgMetadataDict]):
        """ """
        # assert torch.onnx.is_in_onnx_export()
        assert len(one_img) == len(one_img_metas) == 1

        img, img_metas = one_img, one_img_metas

        features = self.extract_feat(img)
        bbox_outs = self.bbox_head(features)
        classification_results = {
            name: head.forward_activate(features)
            for name, head in self.classifier_heads.items()
        }

        img_shape = torch._shape_as_tensor(img)[2:]  # Gets (H, W)
        img_metas[0]["img_shape_for_onnx"] = img_shape
        bbox_list = self.bbox_head.get_bboxes(*bbox_outs, img_metas, rescale=False)

        return bbox_list, list(classification_results.values())

    def _parse_losses(
        self, losses: Dict[str, Union[Tensor, TensorDict, TensorList]]
    ) -> tuple:
        # TODO: Pass weights into loss
        # NOTE: This is where you can pass in weights for each loss function
        r"""Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, coming typically from `self.train_step`

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor \
                which may be a weighted sum of all losses, log_vars contains \
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            elif isinstance(loss_value, dict):
                # Unroll classification losses returned as a dict
                for k, v in loss_value.items():
                    log_vars[f"loss_classification_{k}"] = v
            else:
                raise TypeError(
                    f"{loss_name} is not a tensor or list or dict of tensors"
                )

        loss = sum(_value for _key, _value in log_vars.items() if "loss" in _key)

        log_vars["loss"] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars
