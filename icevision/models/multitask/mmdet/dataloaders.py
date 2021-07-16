# from icevision.all import *
from icevision.imports import *
from icevision.core import *
from icevision.models.multitask.utils.dtypes import *
from icevision.models.multitask.utils.dtypes import (
    DataDictClassification,  # Not imported in __all__ as they are mmdet specific
    DataDictDetection,
)
from icevision.models.mmdet.common.utils import convert_background_from_zero_to_last
from icevision.models.utils import unload_records
from icevision.models.mmdet.common.bbox.dataloaders import (
    _img_tensor,
    _img_meta,
    _labels,
    _bboxes,
)
from icevision.models.multitask.data.dataloading_utils import *
from collections import defaultdict


def build_multi_aug_batch(
    records: Sequence[RecordType], classification_transform_groups: dict
) -> Tuple[
    Dict[str, Union[DataDictClassification, DataDictDetection]], Sequence[RecordType]
]:
    """
    Docs:
        Take as inputs `records` and `classification_transform_groups` and return
        a tuple of dictionaries, one for detection data and the other for classification.

        See `icevision.models.multitask.data.dataset.HybridAugmentationsRecordDataset`
        for example of what `records` and `classification_transform_groups` look like

    Returns:
        A nested data dictionary - (`detection_data`, `classification_data`) and
        the loaded records
        {
            `detection_data`:
                {
                    "detection": dict(
                        images: Tensor = ...,
                        img_metas: Dict[
                            'img_shape': HWC tuple,
                            'pad_shape': HWC tuple,
                            'scale_factor': np.ndarray <len=4>
                        ] = ...,
                        gt_bboxes: Tensor = ...,
                        gt_bbox_labels: Tensor = ...,
                    )
                }

            `classification_data`:
                {
                    "group1": dict(
                        tasks = ["shot_composition"],
                        images: Tensor = ...,
                        classification_labels=dict(
                            "shot_composition": Tensor = ...,
                        )
                    ),
                    "group2": dict(
                        tasks = ["color_saturation", "shot_framing"],
                        images: Tensor = ...,
                        classification_labels=dict(
                            "color_saturation": Tensor = ...,
                            "shot_framing": Tensor = ...,
                        )
                    )
                }
        }
    """
    # NOTE: `detection` is ALWAYS treated as a distinct group
    det_images, bbox_labels, bboxes, img_metas = [], [], [], []
    classification_data = defaultdict(lambda: defaultdict(list))
    classification_labels = defaultdict(list)

    for record in records:
        # Create detection data
        det_images.append(_img_tensor(record.detection))
        img_metas.append(_img_meta(record))
        bbox_labels.append(_labels(record))
        bboxes.append(_bboxes(record))

        # Get classification images for each group
        for key, group in classification_transform_groups.items():
            task = getattr(record, group["tasks"][0])
            # assert (record.color_saturation.img == record.shot_framing.img).all()

            classification_data[key]["tasks"] = group["tasks"]
            classification_data[key]["images"].append(_img_tensor(task))

        # Get classification labels for each group
        for comp in record.components:
            name = comp.task.name
            if isinstance(comp, ClassificationLabelsRecordComponent):
                if comp.is_multilabel:
                    labels = comp.one_hot_encoded()
                    classification_labels[name].append(labels)
                else:
                    labels = comp.label_ids
                    classification_labels[name].extend(labels)

    # Massage data
    for group in classification_data.values():
        group["classification_labels"] = {
            task: tensor(classification_labels[task]) for task in group["tasks"]
        }
        group["images"] = torch.stack(group["images"])
    classification_data = {k: dict(v) for k, v in classification_data.items()}

    detection_data = {
        "img": torch.stack(det_images),
        "img_metas": img_metas,
        "gt_bboxes": bboxes,
        "gt_bbox_labels": bbox_labels,
    }

    data = dict(detection=detection_data, classification=classification_data)
    return data, records


@unload_records
def build_single_aug_batch(records: Sequence[RecordType]):
    """
    Regular `mmdet` dataloader but with classification added in
    """
    images, bbox_labels, bboxes, img_metas = [], [], [], []
    classification_labels = defaultdict(list)

    for record in records:
        images.append(_img_tensor(record))
        img_metas.append(_img_meta(record))
        bbox_labels.append(_labels(record))
        bboxes.append(_bboxes(record))

        # Loop through and create classifier dict of inputs
        for comp in record.components:
            name = comp.task.name
            if isinstance(comp, ClassificationLabelsRecordComponent):
                if comp.is_multilabel:
                    labels = comp.one_hot_encoded()
                    classification_labels[name].append(labels)
                else:
                    labels = comp.label_ids
                    classification_labels[name].extend(labels)

    classification_labels = {k: tensor(v) for k, v in classification_labels.items()}

    data = {
        "img": torch.stack(images),
        "img_metas": img_metas,
        "gt_bboxes": bboxes,
        "gt_bbox_labels": bbox_labels,
        "gt_classification_labels": classification_labels,
    }

    return data, records
