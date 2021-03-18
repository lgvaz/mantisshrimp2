from icevision.imports import *
from icevision import BBox, BaseRecord

# TODO: remove this once DetectedBBox is merged or new Record is used
try:
    from icevision import DetectedBBox
except ImportError:

    class DetectedBBox(BBox):
        def __init__(self, xmin, ymin, xmax, ymax, score, label):
            if score > 1.0:
                raise RuntimeWarning(
                    f"setting detection score larger than 1.0: {score}"
                )
            if score < 0.0:
                raise RuntimeWarning(f"setting detection score lower than 0.0: {score}")
            self.score = score
            self.label = label
            super(DetectedBBox, self).__init__(xmin, ymin, xmax, ymax)

        @classmethod
        def from_xywh(cls, x, y, w, h, score, label):
            bbox = BBox.from_xywh(x, y, w, h)
            return cls(*bbox.xyxy, score, label)


def zeroify_items_below_threshold(
    iou_scores: torch.Tensor, threshold: float
) -> torch.Tensor:
    return iou_scores * (iou_scores > threshold).byte()


def couple_with_targets(predicted_bboxes, iou_scores) -> Sequence:
    """Connects detected bounding boxes with ground truths by iou > 0"""
    ious_per_target = iou_scores.split(1, dim=1)
    return [
        list(itertools.compress(predicted_bboxes, iou.bool()))
        for iou in ious_per_target
    ]


def pick_best_score_labels(predicted_bboxes, confidence_threshold: float = 0.5):
    # fill with dummy if list of predicted labels is empty
    BACKGROUND_IDX = 0
    dummy = DetectedBBox(0, 0, 0, 0, score=1.0, label=BACKGROUND_IDX)
    best_labels = []
    # pick the label that fits best given ground truth
    for ground_truth_predictions in predicted_bboxes:
        ground_truth_predictions = [
            prediction if prediction.score > confidence_threshold else dummy
            for prediction in ground_truth_predictions
        ]
        best_prediction = max(
            ground_truth_predictions, key=lambda x: x.score, default=dummy
        )
        best_labels.append(best_prediction.label)
    return best_labels


def pairwise_bboxes_iou(
    predicted_bboxes: Sequence[BBox], target_bboxes: Sequence[BBox]
):
    """
    Calculates pairwise iou on lists of bounding boxes. Uses torchvision implementation of `box_iou`.
    :param predicted_bboxes:
    :param target_bboxes:
    :return:
    """
    stacked_preds = [bbox.to_tensor() for bbox in predicted_bboxes]
    stacked_preds = torch.stack(stacked_preds) if stacked_preds else torch.empty(0, 4)

    stacked_targets = [bbox.to_tensor() for bbox in target_bboxes]
    stacked_targets = (
        torch.stack(stacked_targets) if stacked_targets else torch.empty(0, 4)
    )
    return torchvision.ops.box_iou(stacked_preds, stacked_targets)


def pairwise_record_bboxes_iou(predictions: BaseRecord, targets: BaseRecord):
    """
    Calculates pairwise iou on lists of bounding boxes. Uses torchvision implementation of `box_iou`.
    :param predictions:
    :param targets:
    :return:
    """
    stacked_predictions_bboxes = [
        bbox.to_tensor() for bbox in predictions.detection.bboxes
    ]
    stacked_predictions_bboxes = (
        torch.stack(stacked_predictions_bboxes)
        if stacked_predictions_bboxes
        else torch.empty(0, 4)
    )

    stacked_targets = [bbox.to_tensor() for bbox in targets.detection.bboxes]
    stacked_targets = (
        torch.stack(stacked_targets) if stacked_targets else torch.empty(0, 4)
    )
    return torchvision.ops.box_iou(stacked_predictions_bboxes, stacked_targets)


def add_unknown_labels(ground_truths, predictions, class_map):
    """
    Add Missing labels to the class map by turning gt and preds to sets and then checking which idxs are missing
    :param ground_truths:
    :param predictions:
    :param class_map:
    :return class_map with added missing keys:
    """
    ground_truth_idxs = set(ground_truths)
    prediction_idxs = set(predictions)
    class_map_idxs = set(class_map._class2id.values())
    for missing_idx in ground_truth_idxs.union(prediction_idxs) - class_map_idxs:
        class_map.add_name(f"unknown_id_{missing_idx}")
    return class_map


def match_preds_with_targets(
    preds: BaseRecord,
    targets: BaseRecord,
    iou_threshold: float = 0.5,
    confidence_threshold: float = 0.5,
) -> Tuple[BBox, Tuple[BBox, float, int]]:

    iou_threshold = 0.5
    confidence_threshold = 0.5
    target_bboxes = targets.detection.bboxes
    target_labels = targets.detection.labels
    if not target_bboxes:
        return  # continue
    # skip if empty ground_truths

    predicted_bboxes = [
        DetectedBBox(*bbox.xyxy, score=score, label=label)
        for bbox, score, label in zip(
            preds.detection.bboxes, preds.detection.scores, preds.detection.labels
        )
    ]
    # get torchvision iou scores (requires conversion to tensors)
    iou_scores = pairwise_bboxes_iou(predicted_bboxes, target_bboxes)
    # TODO: see what happens if that_match is empty
    that_match = torch.any(iou_scores > iou_threshold, dim=1)
    iou_scores = iou_scores[that_match]
    iou_scores = zeroify_items_below_threshold(iou_scores, threshold=iou_threshold)

    # need to use compress cause list indexing with boolean tensor isn't supported
    predicted_bboxes = list(itertools.compress(predicted_bboxes, that_match))
    predicted_bboxes = couple_with_targets(predicted_bboxes, iou_scores)
    predicted_labels = pick_best_score_labels(
        predicted_bboxes, confidence_threshold=confidence_threshold
    )

    assert len(predicted_labels) == len(target_labels)
    # FIXME: missing scores
    return (list(zip(target_bboxes, target_labels)), predicted_bboxes)
