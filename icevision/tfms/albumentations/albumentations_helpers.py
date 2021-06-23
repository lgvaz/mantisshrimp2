__all__ = ["aug_tfms", "resize", "resize_and_pad"]

import albumentations as A

from icevision.imports import *
from icevision.core import *


def resize(size, ratio_resize=A.LongestMaxSize):
    return ratio_resize(size) if isinstance(size, int) else A.Resize(*size[::-1])


def resize_and_pad(
    size: Union[int, Tuple[int, int]],
    pad: A.DualTransform = partial(
        A.PadIfNeeded, border_mode=cv2.BORDER_CONSTANT, value=[124, 116, 104]
    ),
):
    width, height = (size, size) if isinstance(size, int) else size
    return [resize(size), pad(min_height=height, min_width=width)]


def aug_tfms(
    size: Union[int, Tuple[int, int]],
    presize: Optional[Union[int, Tuple[int, int]]] = None,
    horizontal_flip: Optional[A.HorizontalFlip] = A.HorizontalFlip(),
    shift_scale_rotate: Optional[A.ShiftScaleRotate] = A.ShiftScaleRotate(
        rotate_limit=15,
    ),
    rgb_shift: Optional[A.RGBShift] = A.RGBShift(
        r_shift_limit=10,
        g_shift_limit=10,
        b_shift_limit=10,
    ),
    lightning: Optional[A.RandomBrightnessContrast] = A.RandomBrightnessContrast(),
    blur: Optional[A.Blur] = A.Blur(blur_limit=(1, 3)),
    crop_fn: Optional[A.DualTransform] = partial(A.RandomSizedBBoxSafeCrop, p=0.5),
    pad: Optional[A.DualTransform] = partial(
        A.PadIfNeeded, border_mode=cv2.BORDER_CONSTANT, value=[124, 116, 104]
    ),
) -> List[A.BasicTransform]:
    """Collection of useful augmentation transforms.

    # Arguments
        size: The final size of the image. If an `int` is given, the maximum size of
            the image is rescaled, maintaing aspect ratio. If a `tuple` is given,
            the image is rescaled to have that exact size (width, height).
        presizing: Rescale the image before applying other transfroms. If `None` this
                transform is not applied. First introduced by fastai,this technique is
                explained in their book in [this](https://github.com/fastai/fastbook/blob/master/05_pet_breeds.ipynb)
                chapter (tip: search for "Presizing").
        horizontal_flip: Flip around the y-axis. If `None` this transform is not applied.
        shift_scale_rotate: Randomly shift, scale, and rotate. If `None` this transform
                is not applied.
        rgb_shift: Randomly shift values for each channel of RGB image. If `None` this
                transform is not applied.
        lightning: Randomly changes Brightness and Contrast. If `None` this transform
                is not applied.
        blur: Randomly blur the image. If `None` this transform is not applied.
        crop_fn: Randomly crop the image. If `None` this transform is not applied.
                Use `partial` to saturate other parameters of the class.
        pad: Pad the image to `size`, squaring the image if `size` is an `int`.
            If `None` this transform is not applied. Use `partial` to sature other
            parameters of the class.

    # Returns
        A list of albumentations transforms.
    """

    width, height = (size, size) if isinstance(size, int) else size

    tfms = []
    tfms += [resize(presize, A.SmallestMaxSize) if presize is not None else None]
    tfms += [horizontal_flip, shift_scale_rotate, rgb_shift, lightning, blur]
    # Resize as the last transforms to reduce the number of artificial artifacts created
    if crop_fn is not None:
        crop = crop_fn(height=height, width=width)
        tfms += [A.OneOrOther(crop, resize(size), p=crop.p)]
    else:
        tfms += [resize(size)]
    tfms += [pad(min_height=height, min_width=width) if pad is not None else None]

    tfms = [tfm for tfm in tfms if tfm is not None]

    return tfms
