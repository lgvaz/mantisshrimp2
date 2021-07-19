from icevision.soft_dependencies import *


def test_soft_import():
    has_module = soft_import("collections")
    assert has_module


def test_soft_import_fail():
    has_module = soft_import("non_existent_module")
    assert not has_module


def test_soft_dependencies():
    assert SoftDependencies.check() == {
        "fastai": True,
        "pytorch_lightning": True,
        "albumentations": True,
        "effdet": True,
        "wandb": True,
        "resnest": True,
        "sklearn": True,
        "mmdet": True,
        "yolov5": True,
        "timm": True,
    }
