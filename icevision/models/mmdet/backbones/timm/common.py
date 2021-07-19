__all__ = ["MMDetTimmBase"]

import torch.nn as nn
from timm.models.registry import *
from typing import Tuple, Collection, List
from torch import Tensor


class MMDetTimmBase(nn.Module):
    def __init__(
        self,
        model_name: str = None,
        pretrained: bool = True,  # doesn't matter
        out_indices: Collection[int] = (2, 3, 4),
        norm_eval: bool = True,
    ):

        super().__init__()
        self.model_name = model_name
        self.pretrained = pretrained
        self.out_indices = out_indices
        self.norm_eval = norm_eval
        model_fn = model_entrypoint(self.model_name)
        self.model = model_fn(
            pretrained=self.pretrained, features_only=True, out_indices=out_indices
        )

    def init_weights(self, pretrained=None):
        pass

    def forward(self, x) -> Tuple[Tensor]:  # should return a tuple
        return tuple(self.model(x))
