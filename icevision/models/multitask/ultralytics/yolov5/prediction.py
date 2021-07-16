"""
Largely copied over from `icevision.models.ultralytics.yolov5.prectiction`, but with
classification added
"""

from icevision.models.multitask.utils.model import ForwardType
from icevision.utils.utils import unroll_dict
from icevision.imports import *
from icevision.utils import *
from icevision.core import *
from icevision.data import *
from icevision.models.utils import _predict_from_dl

from icevision.models.multitask.ultralytics.yolov5.dataloaders import *
from icevision.models.ultralytics.yolov5.prediction import (
    convert_raw_predictions as convert_raw_detection_predictions,
)
from icevision.models.multitask.utils.prediction import *


@torch.no_grad()
def _predict_batch(
    model: nn.Module,
    batch: Sequence[Tensor],
    records: Sequence[BaseRecord],
    detection_threshold: float = 0.25,
    nms_iou_threshold: float = 0.45,
    keep_images: bool = False,
    device: Optional[torch.device] = None,
) -> List[Prediction]:
    # device issue addressed on discord: https://discord.com/channels/735877944085446747/770279401791160400/832361687855923250
    if device is not None:
        raise ValueError(
            "For YOLOv5 device can only be specified during model creation, "
            "for more info take a look at the discussion here: "
            "https://discord.com/channels/735877944085446747/770279401791160400/832361687855923250"
        )
    grid = model.model[-1].grid[-1]
    # if `grid.numel() == 1` it means the grid isn't initialized yet and we can't
    # trust it's device (will always be CPU)
    device = grid.device if grid.numel() > 1 else model_device(model)

    batch = batch[0].to(device)
    model = model.eval().to(device)

    (det_preds, _), classif_preds = model(batch, step_type=ForwardType.INFERENCE)
    classification_configs = extract_classifier_pred_cfgs(model)

    return convert_raw_predictions(
        batch=batch,
        raw_detection_preds=det_preds,
        raw_classification_preds=classif_preds,
        records=records,
        classification_configs=classification_configs,
        detection_threshold=detection_threshold,
        nms_iou_threshold=nms_iou_threshold,
        keep_images=keep_images,
    )


def predict(
    model: nn.Module,
    dataset: Dataset,
    detection_threshold: float = 0.25,
    nms_iou_threshold: float = 0.45,
    keep_images: bool = False,
    device: Optional[torch.device] = None,
) -> List[Prediction]:
    batch, records = build_infer_batch(dataset)
    return _predict_batch(
        model=model,
        batch=batch,
        records=records,
        detection_threshold=detection_threshold,
        nms_iou_threshold=nms_iou_threshold,
        keep_images=keep_images,
        device=device,
    )


def predict_from_dl(
    model: nn.Module,
    infer_dl: DataLoader,
    show_pbar: bool = True,
    keep_images: bool = False,
    **predict_kwargs,
):
    return _predict_from_dl(
        predict_fn=_predict_batch,
        model=model,
        infer_dl=infer_dl,
        show_pbar=show_pbar,
        keep_images=keep_images,
        **predict_kwargs,
    )


def convert_raw_predictions(
    batch,
    raw_detection_preds: Tensor,
    raw_classification_preds: TensorDict,
    records: Sequence[BaseRecord],
    classification_configs: dict,
    detection_threshold: float = 0.4,
    nms_iou_threshold: float = 0.6,
    keep_images: bool = False,
):
    preds = convert_raw_detection_predictions(
        batch=batch,
        raw_preds=raw_detection_preds,
        records=records,
        detection_threshold=detection_threshold,
        nms_iou_threshold=nms_iou_threshold,
        keep_images=keep_images,
    )
    for pred, raw_classification_pred in zipsafe(
        preds, unroll_dict(raw_classification_preds)
    ):
        add_classification_components_to_pred_record(
            pred_record=pred.pred,
            classification_configs=classification_configs,
        )
        postprocess_and_add_classification_preds_to_record(
            gt_record=pred.ground_truth,
            pred_record=pred.pred,
            classification_configs=classification_configs,
            raw_classification_pred=raw_classification_pred,
        )

    return preds
