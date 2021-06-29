__all__ = [
    "resnet_param_groups",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnext101_32x8d",
]

from icevision.imports import *
from icevision.utils import *


def _resnet_features(model: nn.Module, out_channels: int):
    # remove last layer (fully-connected)
    modules = list(model.named_children())[:-2]
    features = nn.Sequential(OrderedDict(modules))

    features.out_channels = out_channels
    features.param_groups = MethodType(resnet_param_groups, features)

    return features


def resnet_param_groups(model: nn.Module) -> List[nn.Parameter]:
    layers = []
    layers += [nn.Sequential(model.conv1, model.bn1)]
    layers += [l for name, l in model.named_children() if name.startswith("layer")]

    param_groups = [list(layer.parameters()) for layer in layers]
    check_all_model_params_in_groups2(model, param_groups)

    return param_groups


def resnet18(pretrained: bool = True):
    model = torchvision.models.resnet18(pretrained=pretrained)
    return _resnet_features(model, out_channels=512)


def resnet34(pretrained: bool = True):
    model = torchvision.models.resnet34(pretrained=pretrained)
    return _resnet_features(model, out_channels=512)


def resnet50(pretrained: bool = True):
    model = torchvision.models.resnet50(pretrained=pretrained)
    return _resnet_features(model, out_channels=2048)


def resnet101(pretrained: bool = True):
    model = torchvision.models.resnet101(pretrained=pretrained)
    return _resnet_features(model, out_channels=2048)


def resnet152(pretrained: bool = True):
    model = torchvision.models.resnet152(pretrained=pretrained)
    return _resnet_features(model, out_channels=2048)


def resnext101_32x8d(pretrained: bool = True):
    model = torchvision.models.resnext101_32x8d(pretrained=pretrained)
    return _resnet_features(model, out_channels=2048)
