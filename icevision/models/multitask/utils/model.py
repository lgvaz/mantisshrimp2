from enum import Enum
from torch.nn.modules.batchnorm import _BatchNorm
from torch import nn

__all__ = ["ForwardType", "set_bn_eval"]


class ForwardType(Enum):
    TRAIN_MULTI_AUG = 1
    TRAIN = 2
    EVAL = 3
    INFERENCE = 4
    # EXPORT_ONNX = 5
    # EXPORT_TORCHSCRIPT = 6
    # EXPORT_COREML = 7


# Taken from from https://github.com/fastai/fastai/blob/4decc673ba811a41c6e3ab648aab96dd27244ff7/fastai/callback/training.py#L43-L49
def set_bn_eval(m: nn.Module, use_eval=True) -> None:
    "Set bn layers in eval mode for all recursive, non-trainable children of `m`."
    for l in m.children():
        if isinstance(l, _BatchNorm) and not next(l.parameters()).requires_grad:
            l.eval()
        set_bn_eval(l)
