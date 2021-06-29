__all__ = ["adapted_fastai_learner"]

from icevision.imports import *
from icevision.utils import *
from icevision.metrics import *
from icevision.engines.fastai.imports import *
from icevision.engines.fastai.adapters import *


# TODO: param_groups fix for efficientdet
def adapted_fastai_learner(
    dls: List[Union[DataLoader, fastai.DataLoader]],
    model: nn.Module,
    metrics=None,
    device=None,
    splitter=None,
    **learner_kwargs,
) -> fastai.Learner:
    # convert dataloaders to fastai
    fastai_dls = []
    for dl in dls:
        if isinstance(dl, DataLoader):
            fastai_dl = convert_dataloader_to_fastai(dl)
        elif isinstance(dl, fastai.DataLoader):
            fastai_dl = dl
        else:
            raise ValueError(f"dl type {type(dl)} not supported")

        fastai_dls.append(fastai_dl)

    device = device or fastai.default_device()
    fastai_dls = fastai.DataLoaders(*fastai_dls).to(device)

    # convert metrics to fastai
    metrics = metrics or []
    fastai_metrics = [
        FastaiMetricAdapter(metric) if isinstance(metric, Metric) else metric
        for metric in metrics
    ]

    if splitter == None:
        if hasattr(model, "param_groups"):

            def splitter(model):
                return model.param_groups()

        else:
            raise ValueError(
                "If the parameter `splitter` is not specified, "
                "the model should define a method called `param_groups`"
            )

    learn = fastai.Learner(
        dls=fastai_dls,
        model=model,
        metrics=fastai_metrics,
        splitter=splitter,
        **learner_kwargs,
    )
    learn.freeze()
    return learn
