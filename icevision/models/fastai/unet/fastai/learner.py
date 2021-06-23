__all__ = ["learner"]


from icevision.imports import *
from icevision.engines.fastai import *
from icevision.models.fastai.unet.fastai.callbacks import *


def learner(
    dls: List[Union[DataLoader, fastai.DataLoader]],
    model: nn.Module,
    cbs=None,
    loss_func=fastai.CrossEntropyLossFlat(axis=1),
    **kwargs,
):
    cbs = L(UnetCallback()) + L(cbs)

    learn = adapted_fastai_learner(
        dls=dls,
        model=model,
        cbs=cbs,
        loss_func=loss_func,
        **kwargs,
    )

    return learn
