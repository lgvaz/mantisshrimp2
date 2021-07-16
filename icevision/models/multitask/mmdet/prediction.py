# Modified from icevision.models.mmdet.common.bbox.prediction

from icevision.all import *
from icevision.models.mmdet.common.bbox.prediction import _unpack_raw_bboxes

from ..utils import *


__all__ = [
    "predict",
    "predict_from_dl",
    "convert_raw_prediction",
    "convert_raw_predictions",
    "finalize_classifier_preds",
]

from icevision.imports import *
from icevision.utils import *
from icevision.core import *
from icevision.data import *
from icevision.core.tasks import Task
from icevision.models.utils import _predict_from_dl
from icevision.models.mmdet.common.utils import *
from icevision.models.mmdet.common.bbox.dataloaders import build_infer_batch
from icevision.models.mmdet.common.utils import convert_background_from_last_to_zero
from icevision.models.multitask.utils.prediction import finalize_classifier_preds


@torch.no_grad()
def _predict_batch(
    model: nn.Module,
    batch: Sequence[torch.Tensor],
    records: Sequence[BaseRecord],
    classification_configs: dict,
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
        classification_configs=classification_configs,
        keep_images=keep_images,
        detection_threshold=detection_threshold,
    )


def predict(
    model: nn.Module,
    dataset: Dataset,
    classification_configs: dict,
    detection_threshold: float = 0.5,
    keep_images: bool = False,
    device: Optional[torch.device] = None,
) -> List[Prediction]:
    batch, records = build_infer_batch(dataset)

    return _predict_batch(
        model=model,
        batch=batch,
        records=records,
        classification_configs=classification_configs,
        detection_threshold=detection_threshold,
        keep_images=keep_images,
        device=device,
    )


@torch.no_grad()
def _predict_from_dl(
    predict_fn,
    model: nn.Module,
    infer_dl: DataLoader,
    keep_images: bool = False,
    show_pbar: bool = True,
    **predict_kwargs,
) -> List[Prediction]:
    all_preds = []
    for batch, records in pbar(infer_dl, show=show_pbar):
        preds = predict_fn(
            model=model,
            batch=batch,
            records=records,
            keep_images=keep_images,
            **predict_kwargs,
        )
        all_preds.extend(preds)

    return all_preds


def predict_from_dl(
    model: nn.Module,
    infer_dl: DataLoader,
    # classification_configs: dict,
    show_pbar: bool = True,
    keep_images: bool = False,
    **predict_kwargs,
):
    _predict_batch_fn = partial(_predict_batch, keep_images=keep_images)
    # FIXME `classification_configs` needs to be passed in as **predict_kwargs
    return _predict_from_dl(
        predict_fn=_predict_batch_fn,
        model=model,
        # classification_configs=classification_configs,
        infer_dl=infer_dl,
        show_pbar=show_pbar,
        keep_images=keep_images,
        **predict_kwargs,
    )


def convert_raw_predictions(
    batch,
    raw_preds,
    records: Sequence[BaseRecord],
    classification_configs: dict,
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
    bbox_preds, classification_preds = (
        raw_preds["bbox_results"],
        raw_preds["classification_results"],
    )

    # Convert dicts of sequences into a form that we can iterate over in a for loop
    # A test / infer dataloader will not have "gt_classification_labels" as a key
    if "gt_classification_labels" in batch:
        gt_classification_labels = [
            dict(zip(batch["gt_classification_labels"], t))
            for t in zipsafe(*batch["gt_classification_labels"].values())
        ]
        batch["gt_classification_labels"] = gt_classification_labels
    classification_preds = [
        dict(zip(classification_preds, t))
        for t in zipsafe(*classification_preds.values())
    ]
    batch_list = [dict(zip(batch, t)) for t in zipsafe(*batch.values())]

    return [
        convert_raw_prediction(
            sample=sample,
            raw_bbox_pred=bbox_pred,
            raw_classification_pred=classification_pred,
            classification_configs=classification_configs,
            record=record,
            detection_threshold=detection_threshold,
            keep_image=keep_images,
        )
        for sample, bbox_pred, classification_pred, record in zip(
            batch_list, bbox_preds, classification_preds, records
        )
    ]


def convert_raw_prediction(
    sample,
    raw_bbox_pred: dict,
    raw_classification_pred: TensorDict,
    classification_configs: dict,
    record: BaseRecord,
    detection_threshold: float,
    keep_image: bool = False,
):
    # convert predictions
    raw_bboxes = raw_bbox_pred
    scores, labels, bboxes = _unpack_raw_bboxes(raw_bboxes)

    keep_mask = scores > detection_threshold
    keep_scores = scores[keep_mask]
    keep_labels = labels[keep_mask]
    keep_bboxes = [BBox.from_xyxy(*o) for o in bboxes[keep_mask]]

    keep_labels = convert_background_from_last_to_zero(
        label_ids=keep_labels, class_map=record.detection.class_map
    )

    # TODO: Refactor with functions from `...multitask.utils.prediction`
    pred = BaseRecord(
        [
            FilepathRecordComponent(),
            ScoresRecordComponent(),
            ImageRecordComponent(),
            InstancesLabelsRecordComponent(),
            BBoxesRecordComponent(),
            *[ScoresRecordComponent(Task(task)) for task in classification_configs],
            *[
                ClassificationLabelsRecordComponent(
                    Task(task), is_multilabel=cfg.multilabel
                )
                for task, cfg in classification_configs.items()
            ],
        ]
    )
    pred.detection.set_class_map(record.detection.class_map)
    pred.detection.set_scores(keep_scores)
    pred.detection.set_labels_by_id(keep_labels)
    pred.detection.set_bboxes(keep_bboxes)
    pred.above_threshold = keep_mask

    # TODO: Refactor with functions from `...multitask.utils.prediction`
    for task, classification_pred in raw_classification_pred.items():
        labels, scores = finalize_classifier_preds(
            pred=classification_pred,
            cfg=classification_configs[task],
            record=record,
            task=task,
        )
        pred.set_filepath(record.filepath)
        getattr(pred, task).set_class_map(getattr(record, task).class_map)
        getattr(pred, task).set_scores(scores)
        getattr(pred, task).set_labels(labels)

    if keep_image:
        image = mmdet_tensor_to_image(sample["img"])

        pred.set_img(image)
        record.set_img(image)

    return Prediction(pred=pred, ground_truth=record)
