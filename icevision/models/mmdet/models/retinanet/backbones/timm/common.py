__all__ = [
    "MMDetTimmRetinanetBackboneConfig",
]

import icevision
from icevision.imports import *
from icevision.models.mmdet.utils import *


class MMDetTimmRetinanetBackboneConfig(MMDetTimmBackboneConfig):
    def __init__(self, config_path, backbone_dict, weights_url=None, **kwargs):
        super().__init__(
            model_name="retinanet",
            config_path=config_path,
            backbone_dict=backbone_dict,
            weights_url=weights_url,
            **kwargs
        )
