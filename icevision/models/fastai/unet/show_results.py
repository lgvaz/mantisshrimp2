__all__ = ["show_results", "interp"]

from icevision.imports import *
from icevision.utils import *
from icevision.core import *
from icevision.data import *
from icevision.models.fastai.unet.prediction import *
from icevision.models.base_show_results import *
from icevision.models.fastai.unet.dataloaders import (
    valid_dl,
    infer_dl,
)
from icevision.models.interpretation import Interpretation

from icevision.models.interpretation import _move_to_device
from icevision.core.record_components import LossesRecordComponent


def show_results(
    model: nn.Module,
    dataset: Dataset,
    num_samples: int = 6,
    ncols: int = 3,
    denormalize_fn: Optional[callable] = denormalize_imagenet,
    show: bool = True,
    device: Optional[torch.device] = None,
) -> None:
    return base_show_results(
        predict_fn=predict,
        model=model,
        dataset=dataset,
        num_samples=num_samples,
        ncols=ncols,
        denormalize_fn=denormalize_fn,
        show=show,
        device=device,
    )


def loop_unet(dl, model, losses_stats, device):
    samples_plus_losses = []
    loss_func = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for (x, y), sample in pbar(dl):
            torch.manual_seed(0)
            x, y = _move_to_device(x, y, device)
            preds = model(x)
            loss = loss_func(preds, y)
            loss = {
                "loss_unet": float(loss.cpu().numpy()),
                "loss_total": float(loss.cpu().numpy()),
            }

            for l in losses_stats.keys():
                losses_stats[l].append(loss[l])

            loss_comp = LossesRecordComponent()
            loss_comp.set_losses(loss)
            sample[0].add_component(loss_comp)
            sample[0].set_img(tensor_to_image(x[0]))
            sample[0].segmentation.set_mask_array(
                MaskArray(y[0].detach().cpu().numpy())
            )
            samples_plus_losses.append(sample[0])
    return samples_plus_losses, losses_stats


_LOSSES_DICT = {
    "loss_unet": [],
    "loss_total": [],
}

interp = Interpretation(
    losses_dict=_LOSSES_DICT,
    valid_dl=valid_dl,
    infer_dl=infer_dl,
    predict_from_dl=predict_from_dl,
)

interp._loop = loop_unet
