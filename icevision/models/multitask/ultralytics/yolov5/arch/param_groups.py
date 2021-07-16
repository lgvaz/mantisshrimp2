"""
This file defines how to get parameter groups from the `HybridYOLOV5`
model. It is expected to be used along with the other classes in this
submodule, but is defined in a distinct file for easier referencing
and if one wanted to define a custom param_groups functions
"""

from typing import List
from torch.nn import Parameter
from icevision.utils.utils import flatten
from icevision.utils.torch_utils import check_all_model_params_in_groups2

__all__ = ["ParamGroupsExtension"]


class ParamGroupsExtension:
    """
    Splits the model into distinct parameter groups to pass differential
    learning rates to. Given the structure of the model, you must note
    that the param groups are not returned sequentially. The last returned
    group is the classifier heads, and the second last is bbox head, and you
    may want to apply the same LR to both. The `lr=slice(1e-3)` syntax will not
    work for that and you'd have to manually pass in a sequence of
    `len(param_groups)` (5) learning rates instead

    Param Groups:
        1. Stem - The first conv layer
        2. Backbone - Layers 1:10
        3. Neck - The FPN layers i.e. layers 10:23 (24?)
        4. BBox Head - The `Detect` module, which is the last layer in `self.model`
        5. Classifier Heads
    """

    def param_groups(self) -> List[List[Parameter]]:
        param_groups = [
            flatten(self._get_params_stem()),
            flatten(self._get_params_backbone()),
            flatten(self._get_params_neck()),
            flatten(self._get_params_bbox_head()),
            flatten(self._get_params_classifier_heads()),
        ]
        check_all_model_params_in_groups2(self, param_groups=param_groups)
        return param_groups
