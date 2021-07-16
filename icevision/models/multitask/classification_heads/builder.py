from typing import Dict
from .head import CLASSIFICATION_HEADS, ImageClassificationHead, ClassifierConfig
import torch.nn as nn

__all__ = ["build_classifier_heads", "build_classifier_heads_from_configs"]

# Enter dict of dicts as `cfg`
def build_classifier_heads(configs: Dict[str, Dict[str, dict]]) -> nn.ModuleDict:
    """
    Build classification head from a config which is a dict of dicts.
    A head is created for each key in the input dictionary.

    Expected to be used with `mmdet` models as it uses the
    `CLASSIFICATION_HEADS` registry internally

    Returns:
        a `nn.ModuleDict()` mapping keys from `configs` to classifier heads
    """
    heads = nn.ModuleDict()
    # if configs is not None:
    for name, config in configs.items():
        head = CLASSIFICATION_HEADS.build(config)
        heads.update({name: head})
    return heads


def build_classifier_heads_from_configs(
    configs: Dict[str, ClassifierConfig] = None
) -> nn.ModuleDict:
    """
    Build a `nn.ModuleDict` of `ImageClassificationHead`s from a list of `ClassifierConfig`s
    """
    if configs is None:
        return nn.ModuleDict()

    assert isinstance(configs, dict), f"Expected a `dict`, got {type(configs)}"
    if not all(isinstance(cfg, ClassifierConfig) for cfg in configs.values()):
        raise ValueError(
            f"Expected a `list` of `ClassifierConfig`s \n"
            f"Either one or more elements in the list are not of type `ClassifierConfig`"
        )

    heads = nn.ModuleDict()
    for name, config in configs.items():
        head = ImageClassificationHead.from_config(config)
        heads.update({name: head})
    return heads
