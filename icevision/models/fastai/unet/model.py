__all__ = ["model"]

from icevision.imports import *
from icevision.backbones.backbone_config import BackboneConfig
from icevision.models.torchvision.utils import patch_param_groups


# TODO: img_size and visualization
def model(backbone: BackboneConfig, num_classes: int, img_size, channels_in=3):
    img_size = (img_size, img_size) if isinstance(img_size, int) else img_size[::-1]

    model = fastai.models.unet.DynamicUnet(backbone.backbone, num_classes, img_size)
    patch_param_groups(
        model=model,
        head_layers=[model[1:]],
        backbone_param_groups=backbone.backbone.param_groups(),
    )

    return model
