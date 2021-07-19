__all__ = [
    "MMDetBackboneConfig",
    "MMDetTimmBackboneConfig",
    "mmdet_configs_path",
    "param_groups",
    "param_groups_timm",
    "MMDetBackboneConfig",
    "create_model_config",
    "get_feature_channels",
    "download_weights",
]

from icevision.imports import *
from icevision.utils import *
from icevision.backbones import BackboneConfig
from icevision.models.mmdet.download_configs import download_mmdet_configs
from mmdet.models.detectors import *
from mmdet.models.builder import *
from mmcv import Config, ConfigDict
from mmdet.models.backbones.ssd_vgg import SSDVGG
from copy import copy, deepcopy


mmdet_configs_path = download_mmdet_configs()


class MMDetBackboneConfig(BackboneConfig):
    def __init__(self, model_name, config_path, weights_url):
        self.model_name = model_name
        self.config_path = config_path
        self.weights_url = weights_url
        self.pretrained: bool

    def __call__(self, pretrained: bool = True) -> "MMDetBackboneConfig":
        self.pretrained = pretrained
        return self


class MMDetTimmBackboneConfig(MMDetBackboneConfig):
    def __init__(self, model_name, backbone_dict, config_path=None, weights_url=None):
        if not (model_name):
            raise ValueError(
                "The config must have a model name passed to the `model_name` argument"
            )

        if not ("type" in backbone_dict):
            raise ValueError("Backbone dictionary must have a least a `type` key")

        config_path = mmdet_configs_path / model_name / config_path

        super().__init__(
            model_name=model_name, config_path=config_path, weights_url=weights_url
        )

        if not ("pretrained" in backbone_dict):
            backbone_dict["pretrained"] = True
        if not ("out_indices" in backbone_dict):
            backbone_dict["out_indices"] = (2, 3, 4)
        if not ("norm_eval" in backbone_dict):
            backbone_dict["norm_eval"] = True
        if not ("frozen_stem" in backbone_dict):
            backbone_dict["frozen_stem"] = True
        if not ("frozen_stages" in backbone_dict):
            backbone_dict["frozen_stages"] = 1

        self.backbone_dict = backbone_dict

    def __call__(self, pretrained: bool = True) -> "MMDetTimmBackboneConfig":
        self.pretrained = pretrained
        return self


def get_feature_channels(backbone: MMDetTimmBackboneConfig):
    # build a backbone without loading pretrained weights
    # it's used to only get the features info

    backbone_dict = deepcopy(backbone.backbone_dict)
    backbone_dict["pretrained"] = False
    backbone = build_backbone(cfg=backbone_dict)
    all_feature_channels = [o["num_chs"] for o in list(backbone.model.feature_info)]

    # return all the feature channels corresponding to the out_indices
    indices = backbone_dict["out_indices"]
    feature_channels = [
        chs for i, chs in enumerate(all_feature_channels) if i in indices
    ]

    return feature_channels


def download_weights(
    model_name: str,
    weights_url: str,
    checkpoints_path: Optional[Union[str, Path]] = "checkpoints",
    force_download=False,
):
    save_dir = Path(checkpoints_path) / model_name
    save_dir.mkdir(exist_ok=True, parents=True)

    fname = Path(weights_url).name
    weights_path = save_dir / fname

    if not weights_path.exists() or force_download:
        download_url(url=weights_url, save_path=str(weights_path))

    return weights_path


def create_model_config(
    backbone: MMDetBackboneConfig,
    pretrained: bool = True,
    checkpoints_path: Optional[Union[str, Path]] = "checkpoints",
    force_download=False,
):

    model_name = backbone.model_name
    config_path = backbone.config_path
    weights_url = backbone.weights_url

    cfg = Config.fromfile(config_path)
    # timm backbones
    if isinstance(backbone, MMDetTimmBackboneConfig):
        cfg.model.backbone = ConfigDict(backbone.backbone_dict)
        cfg.model.neck.in_channels = get_feature_channels(backbone)

    # MMDetection backbones
    # download weights
    weights_path = None
    if pretrained and weights_url:
        weights_path = download_weights(model_name, weights_url)

    return cfg, weights_path


def param_groups(model):
    body = model.backbone

    layers = []
    if isinstance(body, SSDVGG):
        layers += [body.features]
        layers += [body.extra, body.l2_norm]
    else:
        layers += [nn.Sequential(body.conv1, body.bn1)]
        layers += [getattr(body, l) for l in body.res_layers]
        layers += [model.neck]

    if isinstance(model, SingleStageDetector):
        layers += [model.bbox_head]
    elif isinstance(model, TwoStageDetector):
        layers += [nn.Sequential(model.rpn_head, model.roi_head)]
    else:
        raise RuntimeError(
            "{model} must inherit either from SingleStageDetector or TwoStageDetector class"
        )

    _param_groups = [list(layer.parameters()) for layer in layers]
    check_all_model_params_in_groups2(model, _param_groups)
    return _param_groups


def param_groups_timm(model):
    layers = []

    layers = [model.backbone]
    layers += [model.neck]

    if isinstance(model, SingleStageDetector):
        layers += [model.bbox_head]
    elif isinstance(model, TwoStageDetector):
        layers += [nn.Sequential(model.rpn_head, model.roi_head)]
    else:
        raise RuntimeError(
            "{model} must inherit either from SingleStageDetector or TwoStageDetector class"
        )

    _param_groups = [list(layer.parameters()) for layer in layers]
    check_all_model_params_in_groups2(model, _param_groups)
    return _param_groups
