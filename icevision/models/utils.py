__all__ = [
    "filter_params",
    "unfreeze",
    "freeze",
    "transform_dl",
    "common_build_batch",
    "_predict_dl",
]

from icevision.imports import *
from icevision.utils import *
from icevision.core import *
from icevision.data import *
from icevision.parsers import *

BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)


def filter_params(
    module: nn.Module, bn: bool = True, only_trainable=False
) -> Generator:
    """Yields the trainable parameters of a given module.

    Args:
        module: A given module
        bn: If False, don't return batch norm layers

    Returns:
        Generator
    """
    children = list(module.children())
    if not children:
        if not isinstance(module, BN_TYPES) or bn:
            for param in module.parameters():
                if not only_trainable or param.requires_grad:
                    yield param
    else:
        for child in children:
            for param in filter_params(
                module=child, bn=bn, only_trainable=only_trainable
            ):
                yield param


def unfreeze(params):
    for p in params:
        p.requires_grad = True


def freeze(params):
    for p in params:
        p.requires_grad = False


def _collate_fn(records, build_batch, batch_tfms):
    batch, records = build_batch(records, batch_tfms=batch_tfms)

    for record in records:
        record.unload()

    return batch, records


def transform_dl(dataset, build_batch, batch_tfms=None, **dataloader_kwargs):
    collate_fn = partial(_collate_fn, build_batch=build_batch, batch_tfms=batch_tfms)
    return DataLoader(dataset=dataset, collate_fn=collate_fn, **dataloader_kwargs)


def common_build_batch(records: Sequence[RecordType], batch_tfms=None):
    if batch_tfms is not None:
        records = batch_tfms(records)

    return records


@torch.no_grad()
def _predict_dl(
    predict_fn,
    model: nn.Module,
    infer_dl: DataLoader,
    show_pbar: bool = True,
    **predict_kwargs,
) -> List[Prediction]:
    all_preds = []
    for batch, records in pbar(infer_dl, show=show_pbar):
        preds = predict_fn(model=model, batch=batch, records=records, **predict_kwargs)
        all_preds.extend(preds)

    return all_preds
