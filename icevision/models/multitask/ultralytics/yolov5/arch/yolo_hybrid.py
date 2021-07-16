"""
Multitask implementation of YOLO-V5.
Supports the following tasks:
    * Object Detection
    * Image Classification

See https://discord.com/channels/735877944085446747/770279401791160400/853698548750745610
  for a more detailed discussion
"""

__all__ = ["HybridYOLOV5", "ClassifierConfig"]


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pathlib import Path
from torch import Tensor
from torch.nn.parameter import Parameter
from icevision.models.multitask.classification_heads.head import (
    ClassifierConfig,
    ImageClassificationHead,
    Passthrough,
)
from icevision.models.multitask.utils.dtypes import *
from icevision.models.multitask.classification_heads.builder import (
    build_classifier_heads_from_configs,
)
from icevision.models.multitask.utils.model import ForwardType, set_bn_eval
from icevision.models.multitask.ultralytics.yolov5.arch.model_freezing import *
from icevision.models.multitask.ultralytics.yolov5.arch.param_groups import *

# from .yolo import *
from yolov5.models.yolo import *

from typing import Collection, Dict, Optional, List, Tuple, Union
from copy import deepcopy
from loguru import logger

logger = logger.opt(colors=True)


# fmt: off
YOLO_FEATURE_MAP_DIMS = {
    # models/*yaml
    "yolov5s": [128, 256,  512],            # (128, 32, 32), (256, 16, 16),  (512, 8, 8)
    "yolov5m": [192, 384,  768],            # (192, 32, 32), (384, 16, 16),  (768, 8, 8)
    "yolov5l": [256, 512, 1024],            # (256, 32, 32), (512, 16, 16), (1024, 8, 8)
    "yolov5x": [320, 640, 1280],            # (320, 32, 32), (640, 16, 16), (1280, 8, 8)

    # models/hub/*yaml
    "yolov3-spp":   [256, 512, 1024],             # (256, 32, 32), (512, 16, 16), (1024, 8, 8)
    "yolov3-tiny":  [256, 512],                   # (256, 16, 16), (512,  8,  8)
    "yolov3":       [256, 512, 1024],             # (256, 32, 32), (512, 16, 16), (1024, 8, 8)
    "yolov5-fpn":   [256, 512, 1024],             # (256, 32, 32), (512, 16, 16), (1024, 8, 8)
    "yolov5-p2":    [256, 512, 1024],             # (256, 32, 32), (512, 16, 16), (1024, 8, 8)
    "yolov5-p6":    [256, 512, 768, 1024],        # (256, 32, 32), (512, 16, 16),  (768, 8, 8), (1024, 4, 4)
    "yolov5-p7":    [256, 512, 768, 1024, 1280],  # (256, 32, 32), (512, 16, 16),  (768, 8, 8), (1024, 4, 4), (1280, 2, 2)
    "yolov5-panet": [256, 512, 1024],             # (256, 32, 32), (512, 16, 16), (1024, 8, 8)
    "yolov5l6":     [256, 512, 768, 1024],        # (256, 32, 32), (512, 16, 16),  (768, 8, 8), (1024, 4, 4)
    "yolov5m6":     [192, 384, 576, 768],         # (192, 32, 32), (384, 16, 16),  (576, 8, 8),  (768, 4, 4)
    "yolov5s6":     [128, 256, 384, 512],         # (128, 32, 32), (256, 16, 16),  (384, 8, 8),  (512, 4, 4)
    "yolov5x6":     [320, 640, 960, 1280],        # (320, 32, 32), (640, 16, 16),  (960, 8, 8), (1280, 4, 4)
    "yolov5s-transformer": [128, 256, 512],       # (128, 32, 32), (256, 16, 16),  (512, 8, 8)
}
# fmt: on


class HybridYOLOV5(
    nn.Module,
    FreezingInterfaceExtension,
    ParamGroupsExtension,
):
    """
    Info:
        Create a multitask variant of any YOLO model from ultralytics
        Currently, multitasking detection + classification is supported. An
          arbitrary number of classification heads can be created by passing
          in a dictionary of `ClassifierConfig`s where the keys are names of tasks

    Sample Usage:
        HybridYOLOV5(
            cfg="models/yolov5s.yaml",
            classifier_configs=dict(
                classifier_head_1=ClassifierConfig(out_classes=10),
                classifier_head_2=ClassifierConfig(out_classes=20, multilabel=True),
            ),
        )
    """

    # HACK sort of... as subclassing is a bit problematic with super(...).__init__()
    fuse = Model.fuse
    nms = Model.nms
    _initialize_biases = Model._initialize_biases
    _print_biases = Model._print_biases
    autoshape = Model.autoshape
    info = Model.info

    def __init__(
        self,
        cfg,  # Path to `.yaml` config
        ch=3,  # Num. input channels (3 for RGB image)
        nc=None,  # Num. bbox classes
        anchors=None,
        classifier_configs: Dict[str, ClassifierConfig] = None,
    ):
        super(HybridYOLOV5, self).__init__()

        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub

            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.safe_load(f)  # model dict

        # Define model
        ch = self.yaml["ch"] = self.yaml.get("ch", ch)  # input channels
        if nc and nc != self.yaml["nc"]:
            logger.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc  # override yaml value
        if anchors:
            logger.info(f"Overriding model.yaml anchors with anchors={anchors}")
            self.yaml["anchors"] = round(anchors)  # override yaml value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])
        self.names = [str(i) for i in range(self.yaml["nc"])]  # default names
        self.inplace = self.yaml.get("inplace", True)
        # logger.info([x.shape for x in self.forward(torch.zeros(1, ch, 64, 64))])

        self.classifier_configs = classifier_configs
        self.build_classification_modules()
        self.post_layers_init()

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            # NOTE: This is the only modified line before classifier heads
            # because we are now returning 2 outputs, not one
            m.stride = torch.tensor(
                # Index into [0] because [1]th index is the classification preds
                [s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))[0]]
                # [s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))]
            )  # forward
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()  # only run once
            # logger.info('Strides: %s' % m.stride.tolist())

        self.fpn_dims = YOLO_FEATURE_MAP_DIMS[Path(self.yaml_file).stem]
        # self.fpn_dims = self.extract_features(torch.rand(1, 3, 224, 224))
        self.num_fpn_dims = len(self.fpn_dims)

        # Init weights, biases
        initialize_weights(self)
        self.info()
        logger.success(f"Built *{Path(self.yaml_file).stem}* model successfully")

        self.post_init()

    def post_layers_init(self):
        """
        Run before doing test forward passes for determining the `Detect` (bbox_head) hparams.
        If you want to inject custom modules into the model, this is the place to do it
        """
        pass

    def post_init(self):
        pass

    def train(self, mode: bool = True):
        "Set model to training mode, while freezing non trainable layers' BN statistics"
        super(HybridYOLOV5, self).train(mode)
        set_bn_eval(self)
        return self

    @property
    def num_bbone_blocks(self) -> int:
        return len(self.yaml["backbone"]) - 1

    @property
    def bbone_blocks_start_idx(self) -> int:
        return 1

    @property
    def bbone_blocks_end_idx(self) -> int:
        return len(self.yaml["backbone"])

    def build_classification_modules(self, verbose: bool = True):
        """
        Description:
            Build classifier heads from `self.classifier_configs`.
            Does checks to see if `num_fpn_features` are given and if they are
              correct for each classifier config, and corrects them if not
        """
        arch = Path(self.yaml_file).stem
        fpn_dims = np.array(YOLO_FEATURE_MAP_DIMS[arch])

        for task, cfg in self.classifier_configs.items():
            num_fpn_features = (
                sum(fpn_dims) if cfg.fpn_keys is None else sum(fpn_dims[cfg.fpn_keys])
            )

            if cfg.num_fpn_features is None:
                cfg.num_fpn_features = num_fpn_features

            elif cfg.num_fpn_features != num_fpn_features:
                if verbose:
                    logger.warning(
                        f"Incompatible `num_fpn_features={cfg.num_fpn_features}` detected in task '{task}'. "
                        f"Replacing with the correct dimensions: {num_fpn_features}"
                    )
                cfg.num_fpn_features = num_fpn_features

        self.classifier_heads = build_classifier_heads_from_configs(
            self.classifier_configs
        )
        if verbose:
            logger.success(f"Built classifier heads successfully")

    def forward(
        self,
        x: Union[Tensor, dict],
        profile=False,
        # forward_detection: bool = True,
        # forward_classification: bool = True,
        # activate_classification: bool = False,
        step_type=ForwardType.TRAIN,
    ) -> Tuple[Union[Tensor, TensorList], TensorDict]:
        "Forward method that is dispatched based on `step_type`"

        if step_type is ForwardType.TRAIN or step_type is ForwardType.EVAL:
            # Assume that model is set to `.eval()` mode before calling this function...?
            return self.forward_once(x, profile=profile)

        elif step_type is ForwardType.INFERENCE:
            return self.forward_inference(x)

        elif step_type is ForwardType.TRAIN_MULTI_AUG:
            return self.forward_multi_augment(x)

        else:
            raise RuntimeError(
                f"Invalid `step_type`. Received: {type(step_type.__class__)}; Expected: {ForwardType.__class__}"
            )

    def forward_inference(self, x):
        # You may export model in training mode?
        if not self.training:
            (det_out, _), clf_out = self.forward_once(x, activate_classification=True)
        if self.training:
            self.classifier_heads.eval()  # Turn off dropout
            det_out, clf_out = self.forward_once(x, activate_classification=True)
        return det_out, tuple(clf_out.values())

    # This is here for API compatibility with the main repo; will likely not be used
    def forward_augment(self, x):
        raise NotImplementedError

    def extract_features(self, x: Tensor):
        return self.forward_once(
            x, forward_detection=False, forward_classification=False
        )[0]

    def forward_multi_augment(self, data: dict) -> Tuple[TensorList, TensorDict]:
        """
        Description:
            Multi augmentation training where we do multiple forward passes over the
            same batch, going through different parts of the network each time.

            Detection and classification are treated separately, and within classification,
            you can group together different tasks. A `group` has multiple `tasks`, so we
            extract features once per group, then iterate over each head for that group's
            `tasks`, and compute the outputs from these features

        Args:
            data (dict): Input container with the following structure:
            ```python
            xb = torch.zeros(1, 3, 224, 224)
            multi_aug_data = dict(
                detection={"images": xb},
                classification={
                    "group_1": dict(
                        tasks=["framing", "saturation"],
                        images=x,
                    )
                }
            )
            ```
            Each group in data["classification"]'s `tasks` must correspond to
            a key in `self.classifier_heads`

        Raises:
            RuntimeError: If model is not in `training` mode (as a safety check)

        Returns:
            Tuple[TensorList, TensorDict]: Tuple of `(detection_preds, classification_preds)`
        """
        if not self.training:
            raise RuntimeError(f"Can only run `forward_multi_augment` in training mode")

        # Detection forward pass
        xb = data["detection"]["images"]
        detection_preds, _ = self.forward_once(
            xb, forward_detection=True, forward_classification=False
        )

        # Classification forward pass
        classification_preds = {}
        for group, datum in data["classification"].items():
            xb = datum["images"]
            features, _ = self.forward_once(
                xb, forward_detection=False, forward_classification=False
            )
            for name in datum["tasks"]:
                head = self.classifier_heads[name]
                classification_preds[name] = head(features)

        return detection_preds, classification_preds

    def forward_once(
        self,
        x,
        profile=False,  # Will fail
        forward_detection: bool = True,
        forward_classification: bool = True,
        activate_classification: bool = False,
    ) -> Tuple[Union[TensorList, Tuple[Tensor, TensorList]], TensorDict]:
        """
        Returns:
            A tuple of two elements `(detection_preds, classification_preds)`:
            1) A TensorList in training mode, and a Tuple[Tensor, TensorList] in
            eval mode where the first element (Tensor) is the inference output and
            second is the training output (for loss computation). If `forward_detection` is
            False, the list of FPN features are returned right before feeding into the bbox
            head i.e. the `Detect` module which can be accessed via `self.model[-1]`
            2) A TensorDict of classification predictions. If `forward_classification` is
            False, an empty dictionary is returned
        """
        y, dt = [], []  # outputs
        classification_preds: Dict[str, Tensor] = {}
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = (
                    y[m.f]
                    if isinstance(m.f, int)
                    else [x if j == -1 else y[j] for j in m.f]
                )  # from earlier layers

            if profile:
                o = (
                    thop.profile(m, inputs=(x,), verbose=False)[0] / 1e9 * 2
                    if thop
                    else 0
                )  # FLOPs
                t = time_synchronized()
                for _ in range(10):
                    _ = m(x)
                dt.append((time_synchronized() - t) * 100)
                if m == self.model[0]:
                    logger.info(
                        f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  {'module'}"
                    )
                logger.info(f"{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}")

            """
            This is where the feature maps are passed into the classification heads.
            Is there a cleaner way to do this? It's tricky as the whole model is wrapped in an
              `nn.Sequential` container and we can't access attribues like `.backbone` or `.neck`.
            We know for certain that `Detect` is the last layer in the model, so this should be
              safe to do.
            """
            if isinstance(m, Detect):
                if forward_classification:
                    for name, head in self.classifier_heads.items():
                        classification_preds[name] = (
                            head.forward_activate(x)
                            if activate_classification
                            else head(x)
                        )

                if not forward_detection:
                    if profile:
                        logger.info("%.1fms total" % sum(dt))
                    return x, classification_preds

            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output

        if profile:
            logger.info("%.1fms total" % sum(dt))

        return x, classification_preds
