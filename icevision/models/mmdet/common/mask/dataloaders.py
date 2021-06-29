__all__ = [
    "build_train_batch",
    "build_valid_batch",
    "build_infer_batch",
    "train_dl",
    "valid_dl",
    "infer_dl",
]

from mmdet.core import BitmapMasks
from icevision.core import *
from icevision.imports import *
from icevision.models.utils import *
from icevision.models.mmdet.common.bbox.dataloaders import (
    _img_tensor,
    _img_meta,
    _labels,
    _bboxes,
)


def train_dl(dataset, batch_tfms=None, **dataloader_kwargs) -> DataLoader:
    return transform_dl(
        dataset=dataset,
        build_batch=build_train_batch,
        batch_tfms=batch_tfms,
        **dataloader_kwargs
    )


def valid_dl(dataset, batch_tfms=None, **dataloader_kwargs) -> DataLoader:
    return transform_dl(
        dataset=dataset,
        build_batch=build_valid_batch,
        batch_tfms=batch_tfms,
        **dataloader_kwargs
    )


def infer_dl(dataset, batch_tfms=None, **dataloader_kwargs) -> DataLoader:
    """A `DataLoader` with a custom `collate_fn` that batches items as required for inferring the model.

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
        build_batch=build_infer_batch,
        batch_tfms=batch_tfms,
        **dataloader_kwargs
    )


def build_valid_batch(
    records: Sequence[RecordType],
) -> Tuple[dict, List[Dict[str, torch.Tensor]]]:
    return build_train_batch(records=records)


def build_train_batch(
    records: Sequence[RecordType],
) -> Tuple[dict, List[Dict[str, torch.Tensor]]]:
    images, labels, bboxes, masks, img_metas = [], [], [], [], []
    for record in records:
        images.append(_img_tensor(record))
        img_metas.append(_img_meta_mask(record))
        labels.append(_labels(record))
        bboxes.append(_bboxes(record))
        masks.append(_masks(record))

    data = {
        "img": torch.stack(images),
        "img_metas": img_metas,
        "gt_labels": labels,
        "gt_bboxes": bboxes,
        "gt_masks": masks,
    }

    return data, records


def build_infer_batch(records):
    imgs, img_metas = [], []
    for record in records:
        imgs.append(_img_tensor(record))
        img_metas.append(_img_meta_mask(record))

    data = {
        "img": [torch.stack(imgs)],
        "img_metas": [img_metas],
    }

    return data, records


def _img_meta_mask(record):
    img_meta = _img_meta(record)
    img_meta["ori_shape"] = img_meta["pad_shape"]
    return img_meta


def _masks(record):
    if len(record.detection.masks) == 0:
        raise RuntimeError("Negative samples still needs to be implemented")
    else:
        mask = record.detection.mask_array.data
        _, h, w = mask.shape
        return BitmapMasks(mask, height=h, width=w)
