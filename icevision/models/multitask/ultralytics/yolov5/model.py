"""
Largely copied over from `icevision.models.ultralytics.yolov5.model`
The only aspect added is the ability to pass in a `Dict[str, ClassifierCongig]` to
  create the classification heads
"""


__all__ = ["model"]

from icevision.imports import *
from icevision.utils import *

import yaml
import yolov5
from yolov5.utils.google_utils import attempt_download
from yolov5.utils.torch_utils import intersect_dicts
from yolov5.utils.general import check_img_size

# from icevision.models.ultralytics.yolov5.utils import *
from icevision.models.multitask.ultralytics.yolov5.utils import *
from icevision.models.ultralytics.yolov5.backbones import *

from icevision.models.multitask.ultralytics.yolov5.arch.yolo_hybrid import HybridYOLOV5
from icevision.models.multitask.classification_heads import ClassifierConfig

yolo_dir = get_root_dir() / "yolo"
yolo_dir.mkdir(exist_ok=True)


def model(
    backbone: YoloV5BackboneConfig,
    num_detection_classes: int,
    img_size: int,  # must be multiple of 32
    device: Optional[torch.device] = None,
    classifier_configs: Dict[str, ClassifierConfig] = None,
) -> HybridYOLOV5:
    """
    Build a `HybridYOLOV5` Multitask Model with detection & classification heads.

    Args:
        backbone (YoloV5BackboneConfig): Config from `icevision.models.ultralytics.yolov5.backbones.{}`
        num_detection_classes (int): Number of object detection classes (including background)
        img_size (int): Size of input images (assumes square inputs)
        classifier_configs (Dict[str, ClassifierConfig], optional): A dictionary mapping of `ClassifierConfig`s
        where each key corresponds to the name of the task in the input records. Defaults to None.

    Returns:
        HybridYOLOV5: A multitask YOLO-V5 model with one detection head and `len(classifier_configs)`
        classification heads
    """
    model_name = backbone.model_name
    pretrained = backbone.pretrained

    # this is to remove background from ClassMap as discussed
    # here: https://github.com/ultralytics/yolov5/issues/2950
    # and here: https://discord.com/channels/735877944085446747/782062040168267777/836692604224536646
    # so we should pass `num_detection_classes=parser.class_map.num_detection_classes`
    num_detection_classes -= 1

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device is None
        else device
    )

    cfg_filepath = Path(yolov5.__file__).parent / f"models/{model_name}.yaml"
    if pretrained:
        weights_path = yolo_dir / f"{model_name}.pt"

        with open(Path(yolov5.__file__).parent / "data/hyp.finetune.yaml") as f:
            hyp = yaml.load(f, Loader=yaml.SafeLoader)

        attempt_download(weights_path)  # download if not found locally
        sys.path.insert(0, str(Path(yolov5.__file__).parent))
        ckpt = torch.load(weights_path, map_location=device)  # load checkpoint
        sys.path.remove(str(Path(yolov5.__file__).parent))
        if hyp.get("anchors"):
            ckpt["model"].yaml["anchors"] = round(hyp["anchors"])  # force autoanchor
        model = HybridYOLOV5(
            cfg_filepath or ckpt["model"].yaml,
            ch=3,
            nc=num_detection_classes,
            classifier_configs=classifier_configs,
        ).to(device)
        exclude = []  # exclude keys
        state_dict = ckpt["model"].float().state_dict()  # to FP32
        state_dict = intersect_dicts(
            state_dict, model.state_dict(), exclude=exclude
        )  # intersect
        model.load_state_dict(state_dict, strict=False)  # load
    else:
        with open(Path(yolov5.__file__).parent / "data/hyp.scratch.yaml") as f:
            hyp = yaml.load(f, Loader=yaml.SafeLoader)

        model = HybridYOLOV5(
            cfg_filepath,
            ch=3,
            nc=num_detection_classes,
            anchors=hyp.get("anchors"),
            classifier_configs=classifier_configs,
        ).to(device)

    gs = int(model.stride.max())  # grid size (max stride)
    nl = model.model[-1].nl  # number of detection layers (used for scaling hyp['obj'])
    imgsz = check_img_size(img_size, gs)  # verify imgsz are gs-multiples

    hyp["box"] *= 3.0 / nl  # scale to layers
    hyp["cls"] *= num_detection_classes / 80.0 * 3.0 / nl  # scale to classes and layers
    hyp["obj"] *= (imgsz / 640) ** 2 * 3.0 / nl  # scale to image size and layers
    model.nc = num_detection_classes  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)

    return model
