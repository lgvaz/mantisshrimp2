__all__ = ["convert_dataloader_to_fastai"]

from icevision.imports import *
from icevision.engines.fastai.imports import *
from torch.utils.data import SequentialSampler, RandomSampler
from torch.utils.data.dataloader import _InfiniteConstantSampler


def convert_dataloader_to_fastai(dataloader: DataLoader):
    def raise_error_convert(data):
        raise NotImplementedError

    class FastaiDataLoaderWithCollate(fastai.DataLoader):
        def create_batch(self, b):
            return (dataloader.collate_fn, raise_error_convert)[self.prebatched](b)

    # use the type of sampler to determine if shuffle is true or false
    if isinstance(dataloader.sampler, (SequentialSampler, _InfiniteConstantSampler)):
        shuffle = False
    elif isinstance(dataloader.sampler, RandomSampler):
        shuffle = True
    else:
        raise ValueError(
            f"Sampler {type(dataloader.sampler)} not supported. Fastai only"
            "supports RandomSampler or SequentialSampler"
        )

    return FastaiDataLoaderWithCollate(
        dataset=dataloader.dataset,
        bs=dataloader.batch_size,
        num_workers=dataloader.num_workers,
        drop_last=dataloader.drop_last,
        shuffle=shuffle,
        pin_memory=dataloader.pin_memory,
    )
