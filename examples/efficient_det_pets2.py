from mantisshrimp.imports import *
from mantisshrimp import *
from mantisshrimp.engines.fastai import *
import albumentations as A


data_dir = datasets.pets.load()
parser = datasets.pets.parser(data_dir)

splitter = RandomSplitter([0.8, 0.2])
train_records, valid_records = parser.parse(splitter)

imagenet_mean, imagenet_std = IMAGENET_STATS
img_size = 256
valid_tfms = AlbuTransform(
    [
        A.Resize(img_size, img_size, always_apply=True),
        A.Normalize(mean=imagenet_mean, std=imagenet_std),
    ]
)

train_tfms = AlbuTransform(
    [
        # A.LongestMaxSize(img_size),
        # A.RandomSizedBBoxSafeCrop(320, 320, p=0.3),
        A.Resize(img_size, img_size, always_apply=True),
        A.HorizontalFlip(),
        A.ShiftScaleRotate(rotate_limit=20),
        A.RGBShift(always_apply=True),
        A.RandomBrightnessContrast(),
        A.Blur(blur_limit=(1, 3)),
        A.Normalize(mean=imagenet_mean, std=imagenet_std),
    ]
)


train_ds = Dataset(train_records[:100], train_tfms)
valid_ds = Dataset(valid_records[:100], valid_tfms)

train_dl = efficient_det.train_dataloader(train_ds, batch_size=16)
valid_dl = efficient_det.valid_dataloader(valid_ds, batch_size=16)

# Plain pytorch model, no magic
# each model will receive differente parameters, since this is heavily
# dependent on the underlying implementation
model, param_groups = efficient_det.model(
    model_name="efficientdet_d0",
    num_classes=len(datasets.pets.CLASSES),
    img_size=img_size,
    pretrained=True,
)

learn = efficient_det.fastai.learner(
    dls=[train_dl, valid_dl], model=model, param_groups=param_groups
)

learn.fit_one_cycle(10, 1e-3)
