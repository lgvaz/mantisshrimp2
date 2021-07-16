from icevision.imports import *
from icevision.core import *
from icevision.utils import Dictionary
from icevision.models.multitask.classification_heads.head import (
    ImageClassificationHead,
    ClassifierConfig,
    TensorDict,
)
from icevision.core.tasks import Task


# __all__ = ["finalize_classifier_preds"]


def finalize_classifier_preds(
    pred, cfg: Dictionary, record: RecordType, task: str
) -> tuple:
    """
    Analyse preds post-activations based on `cfg` arguments; return the
    relevant scores and string labels derived from `record`

    Can compute the following:
        * top-k (`cfg` defaults to 1 for single-label problems)
        * filter preds by threshold
    """

    # pred = np.array(pred)
    pred = pred.detach().cpu().numpy()

    if cfg.topk is not None:
        index = np.argsort(pred)[-cfg.topk :]  # argsort gives idxs in ascending order
        value = pred[index]

    elif cfg.thresh is not None:
        index = np.where(pred > cfg.thresh)[0]  # index into the tuple
        value = pred[index]

    labels = [getattr(record, task).class_map._id2class[i] for i in index]
    scores = pred[index].tolist()

    return labels, scores


def extract_classifier_pred_cfgs(model: nn.Module):
    return {
        name: Dictionary(multilabel=head.multilabel, topk=head.topk, thresh=head.thresh)
        for name, head in model.classifier_heads.items()
    }


def add_classification_components_to_pred_record(
    pred_record: RecordType, classification_configs: dict
):
    """
    Adds `ClassificationLabelsRecordComponent` and `ScoresRecordComponent` to `pred_record`
    for each task; where the keys of `classification_configs` are the names of the tasks

    Args:
        pred_record (RecordType)
        classification_configs (dict)

    Returns:
        [type]: [description]
    """
    r = pred_record
    for name, cfg in classification_configs.items():
        r.add_component(ScoresRecordComponent(Task(name=name)))
        r.add_component(
            ClassificationLabelsRecordComponent(
                Task(name=name), is_multilabel=cfg.multilabel
            )
        )
    return r


def postprocess_and_add_classification_preds_to_record(
    gt_record: RecordType,
    pred_record: RecordType,
    classification_configs: dict,
    raw_classification_pred: TensorDict,
):
    """
    Postprocesses predictions based on `classification_configs` and adds the results
    to `pred_record`. Uses `gt_record` to set the `pred_record`'s class maps

    Args:
        gt_record (RecordType)
        pred_record (RecordType)

        classification_configs (dict): A dict that describes how to postprocess raw
        classification preds. Note that the raw preds are assumed to have already gone
        through an activation function like Softmax or Sigmoid. For example:
            dict(
                multilabel=False, topk=1, thresh=None
            )

        raw_classification_pred (TensorDict): Container whose preds will be processed. Is
        expected to have the exact same keys as `classification_configs`
    """
    for task, classification_pred in raw_classification_pred.items():
        labels, scores = finalize_classifier_preds(
            pred=classification_pred,
            cfg=classification_configs[task],
            record=gt_record,
            task=task,
        )
        # sub_record = getattr(pred_record, task)
        getattr(pred_record, task).set_class_map(getattr(gt_record, task).class_map)
        getattr(pred_record, task).set_labels(labels)
        getattr(pred_record, task).set_scores(scores)
