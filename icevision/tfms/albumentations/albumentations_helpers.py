__all__ = ["aug_tfms", "resize", "resize_and_pad", "get_size_without_padding"]

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


def get_size_without_padding(
    tfms_list: List[Any], before_tfm_img: PIL.Image.Image, height: int, width: int
) -> Tuple[int, int]:
    """
    Infer the height and width of the pre-processed image after removing padding.

    Parameters
    ----------
    tfms_list: list of albumentations transforms applied to the `before_tfm_img` image
                before passing it to the model for inference.
    before_tfm_img: original image before being pre-processed for inference.
    height: height of output image from icevision `predict` function.
    width: width of output image from icevision `predict` function.

    Returns
    -------
    height and width of the image coming out of the inference pipeline, after removing padding
    """
    if get_transform(tfms_list, "Pad") is not None:
        before_pad_h, before_pad_w, _ = np.array(before_tfm_img).shape

        t = get_transform(tfms_list, "SmallestMaxSize")
        if t is not None:
            presize = t.max_size
            height, width = func_max_size(before_pad_h, before_pad_w, presize, min)

        t = get_transform(tfms_list, "LongestMaxSize")
        if t is not None:
            size = t.max_size
            height, width = func_max_size(before_pad_h, before_pad_w, size, max)

    return height, width


def py3round(number: float) -> int:
    """
    Unified rounding in all python versions. Used by albumentations.

    Parameters
    ----------
    number: float to round.

    Returns
    -------
    Rounded number
    """
    if abs(round(number) - number) == 0.5:
        return int(2.0 * round(number / 2.0))

    return int(round(number))


def func_max_size(
    height: int, width: int, max_size: int, func: Callable[[int, int], int]
) -> Tuple[int, int]:
    """
    Calculate rescaled height and width of the image in question wrt to a specific size.

    Parameters
    ----------
    height: height of the image in question.
    width: width of the image in question.
    max_size: size wrt the image needs to be rescaled (resized).
    func: min/max. Whether to compare max_size to the smallest/longest of the image dims.

    Returns
    -------
    Rescaled height and width
    """
    scale = max_size / float(func(width, height))

    if scale != 1.0:
        height, width = tuple(py3round(dim * scale) for dim in (height, width))
    return height, width


def get_transform(tfms_list: List[Any], t: str) -> Any:
    """
    Extract transform `t` from `tfms_list`.

    Parameters
    ----------
    tfms_list: list of albumentations transforms.
    t: name (str) of the transform to look for and return from within `tfms_list`.

    Returns
    -------
    The `t` transform if found inside `tfms_list`, otherwise None.
    """
    for el in tfms_list:
        if t in str(type(el)):
            return el
    return None
