__all__ = ["predict", "predict_from_dl", "convert_raw_predictions"]

from icevision.imports import *
from icevision.utils import *
from icevision.core import *
from icevision.data import *
from icevision.models.utils import _predict_from_dl
from icevision.models.fastai.unet.dataloaders import *


@torch.no_grad()
def _predict_batch(
    model: nn.Module,
    batch: Sequence[torch.Tensor],
    records: Sequence[BaseRecord],
    keep_images: bool = False,
    device: Optional[torch.device] = None,
) -> List[Prediction]:
    device = model_device(model)

    images, *_ = batch
    images = images.to(device)

    raw_preds = model(images)
    preds = convert_raw_predictions(
        batch=batch,
        raw_preds=raw_preds,
        records=records,
        keep_images=keep_images,
    )

    return preds


def convert_raw_predictions(
    batch,
    raw_preds: torch.Tensor,
    records: Sequence[BaseRecord],
    keep_images: bool = False,
) -> List[Prediction]:
    mask_preds = raw_preds.argmax(dim=1)
    tensor_images, *_ = batch

    preds = []
    for record, tensor_image, mask_pred in zip(records, tensor_images, mask_preds):
        pred = BaseRecord(
            (
                ImageRecordComponent(),
                SemanticMaskRecordComponent(),
                ClassMapRecordComponent(task=tasks.segmentation),
            )
        )

        pred.segmentation.set_class_map(record.segmentation.class_map)
        pred.segmentation.set_mask_array(MaskArray(mask_pred.cpu().numpy()[None]))

        if keep_images:
            record.set_img(tensor_to_image(tensor_image))

        preds.append(Prediction(pred=pred, ground_truth=record))

    return preds


def predict(
    model: nn.Module,
    dataset: Dataset,
    keep_images: bool = False,
    device: Optional[torch.device] = None,
) -> List[Prediction]:

    batch, records = build_infer_batch(dataset)

    return _predict_batch(
        model=model,
        batch=batch,
        records=records,
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
