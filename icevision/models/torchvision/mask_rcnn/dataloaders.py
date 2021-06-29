__all__ = [
    "train_dl",
    "valid_dl",
    "infer_dl",
    "build_train_batch",
    "build_valid_batch",
    "build_infer_batch",
]

from icevision.imports import *
from icevision.core import *
from icevision.models.utils import *
from icevision.models.torchvision.faster_rcnn.dataloaders import (
    _build_train_sample,
)
from icevision.models.torchvision.faster_rcnn.dataloaders import (
    build_infer_batch,
    infer_dl,
)


def train_dl(dataset, batch_tfms=None, **dataloader_kwargs) -> DataLoader:
    """A `DataLoader` with a custom `collate_fn` that batches items as required for training the model.

    # Arguments
        dataset: Possibly a `Dataset` object, but more generally, any `Sequence` that returns records.
        batch_tfms: Transforms to be applied at the batch level.
        **dataloader_kwargs: Keyword arguments that will be internally passed to a Pytorch `DataLoader`.
        The parameter `collate_fn` is already defined internally and cannot be passed here.

    # Returns
        A Pytorch `DataLoader`.
    """
    return transform_dl(
        dataset=dataset,
        build_batch=build_train_batch,
        batch_tfms=batch_tfms,
        **dataloader_kwargs
    )


def valid_dl(dataset, batch_tfms=None, **dataloader_kwargs) -> DataLoader:
    """A `DataLoader` with a custom `collate_fn` that batches items as required for validating the model.

    # Arguments
        dataset: Possibly a `Dataset` object, but more generally, any `Sequence` that returns records.
        batch_tfms: Transforms to be applied at the batch level.
        **dataloader_kwargs: Keyword arguments that will be internally passed to a Pytorch `DataLoader`.
        The parameter `collate_fn` is already defined internally and cannot be passed here.

    # Returns
        A Pytorch `DataLoader`.
    """
    return transform_dl(
        dataset=dataset,
        build_batch=build_valid_batch,
        batch_tfms=batch_tfms,
        **dataloader_kwargs
    )


def _build_mask_train_sample(record: RecordType):
    assert (
        len(record.detection.label_ids)
        == len(record.detection.bboxes)
        == len(record.detection.bboxes)
    )

    image, target = _build_train_sample(record=record)

    # If no labels and bboxes are present, use as negative samples as described in
    # https://github.com/pytorch/vision/releases/tag/v0.6.0
    if len(record.detection.masks) == 0:
        height, width = record.img.shape[:-1]
        target["masks"] = torch.zeros((0, height, width), dtype=torch.uint8)
    else:
        target["masks"] = tensor(record.detection.mask_array.data, dtype=torch.uint8)

    return image, target


def build_train_batch(
    records: List[RecordType],
) -> Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]]]:
    """Builds a batch in the format required by the model when training.

    # Arguments
        records: A `Sequence` of records.
        batch_tfms: Transforms to be applied at the batch level.

    # Returns
        A tuple with two items. The first will be a tuple like `(images, targets)`,
        in the input format required by the model. The second will be a list
        of the input records.

    # Examples

    Use the result of this function to feed the model.
    ```python
    batch, records = build_train_batch(records)
    outs = model(*batch)
    ```
    """
    images, targets = [], []
    for record in records:
        image, target = _build_mask_train_sample(record)
        images.append(image)
        targets.append(target)

    return (images, targets), records


def build_valid_batch(
    records: List[RecordType],
) -> Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]]]:
    """Builds a batch in the format required by the model when validating.

    # Arguments
        records: A `Sequence` of records.
        batch_tfms: Transforms to be applied at the batch level.

    # Returns
        A tuple with two items. The first will be a tuple like `(images, targets)`,
        in the input format required by the model. The second will be a list
        of the input records.

    # Examples

    Use the result of this function to feed the model.
    ```python
    batch, records = build_valid_batch(records)
    outs = model(*batch)
    ```
    """
    return build_train_batch(records=records)
