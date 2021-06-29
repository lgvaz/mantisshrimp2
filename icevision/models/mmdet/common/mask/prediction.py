__all__ = [
    "predict",
    "predict_from_dl",
    "convert_raw_prediction",
    "convert_raw_predictions",
]

from icevision.imports import *
from icevision.utils import *
from icevision.core import *
from icevision.data import *
from icevision.models.utils import _predict_from_dl
from icevision.models.mmdet.common.utils import *
from icevision.models.mmdet.common.mask.dataloaders import *
from icevision.models.mmdet.common.bbox.prediction import (
    _unpack_raw_bboxes,
)


@torch.no_grad()
def _predict_batch(
    model: nn.Module,
    batch: Sequence[torch.Tensor],
    records: Sequence[BaseRecord],
    detection_threshold: float = 0.5,
    keep_images: bool = False,
    device: Optional[torch.device] = None,
):
    device = device or model_device(model)
    batch["img"] = [img.to(device) for img in batch["img"]]

    raw_preds = model(return_loss=False, rescale=False, **batch)
    return convert_raw_predictions(
        batch=batch,
        raw_preds=raw_preds,
        records=records,
        keep_images=keep_images,
        detection_threshold=detection_threshold,
    )


def predict(
    model: nn.Module,
    dataset: Dataset,
    detection_threshold: float = 0.5,
    keep_images: bool = False,
    device: Optional[torch.device] = None,
) -> List[Prediction]:
    batch, records = build_infer_batch(dataset)

    return _predict_batch(
        model=model,
        batch=batch,
        records=records,
        detection_threshold=detection_threshold,
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
    raw_preds,
    records: Sequence[BaseRecord],
    detection_threshold: float,
    keep_images: bool = False,
):

    # In inference, both "img" and "img_metas" are lists. Check out the `build_infer_batch()` definition
    # We need to convert that to a batch similar to train and valid batches
    if isinstance(batch["img"], list):
        batch = {
            "img": batch["img"][0],
            "img_metas": batch["img_metas"][0],
        }

    batch_list = [dict(zip(batch, t)) for t in zipsafe(*batch.values())]
    return [
        convert_raw_prediction(
            sample=sample,
            raw_pred=raw_pred,
            record=record,
            detection_threshold=detection_threshold,
            keep_image=keep_images,
        )
        for sample, raw_pred, record in zip(batch_list, raw_preds, records)
    ]


def convert_raw_prediction(
    sample,
    raw_pred: dict,
    record: BaseRecord,
    detection_threshold: float,
    keep_image: bool = False,
):
    # convert predictions
    raw_bboxes, raw_masks = raw_pred
    scores, labels, bboxes = _unpack_raw_bboxes(raw_bboxes)

    keep_mask = scores > detection_threshold
    keep_scores = scores[keep_mask]
    keep_labels = labels[keep_mask]
    keep_bboxes = [BBox.from_xyxy(*o) for o in bboxes[keep_mask]]
    keep_masks = MaskArray(np.vstack(raw_masks)[keep_mask])

    keep_labels = convert_background_from_last_to_zero(
        label_ids=keep_labels, class_map=record.detection.class_map
    )

    pred = BaseRecord(
        (
            ScoresRecordComponent(),
            ImageRecordComponent(),
            InstancesLabelsRecordComponent(),
            BBoxesRecordComponent(),
            InstanceMasksRecordComponent(),
        )
    )
    pred.detection.set_class_map(record.detection.class_map)
    pred.detection.set_scores(keep_scores)
    pred.detection.set_labels_by_id(keep_labels)
    pred.detection.set_bboxes(keep_bboxes)
    pred.detection.set_masks(keep_masks)
    pred.above_threshold = keep_mask

    if keep_image:
        image = mmdet_tensor_to_image(sample["img"])

        pred.set_img(image)
        record.set_img(image)

    return Prediction(pred=pred, ground_truth=record)
