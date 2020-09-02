# The draw functions are based from:
# https://github.com/fizyr/keras-retinanet/blob/master/keras_retinanet/utils/visualization.py
# https://github.com/fizyr/keras-maskrcnn/blob/master/keras_maskrcnn/utils/visualization.py

__all__ = [
    "draw_sample",
    "draw_record",
    "draw_pred",
    "draw_bbox",
    "draw_mask",
]

from mantisshrimp.imports import *
from mantisshrimp.data import *
from mantisshrimp.core import *


def draw_sample(
    sample,
    class_map: Optional[ClassMap] = None,
    denormalize_fn: Optional[callable] = None,
):
    img = sample["img"].copy()
    if denormalize_fn is not None:
        img = denormalize_fn(img)

    for label, bbox, mask in itertools.zip_longest(
        sample.get("labels", []), sample.get("bboxes", []), sample.get("masks", [])
    ):
        color = (np.random.random(3) * 0.6 + 0.4) * 255

        if mask is not None:
            img = draw_mask(img=img, mask=mask, color=color)
        if bbox is not None:
            img = draw_bbox(img=img, bbox=bbox, color=color)
        if label is not None:
            img = draw_label(
                img=img,
                label=label,
                bbox=bbox,
                mask=mask,
                class_map=class_map,
                color=color,
            )

    return img


def draw_label(
    img: np.ndarray,
    label: int,
    color,
    class_map: Optional[ClassMap] = None,
    bbox=None,
    mask=None,
):
    if bbox is not None:
        x, y = bbox.x, bbox.y
    elif mask is not None:
        y, x = np.unravel_index(mask.data.argmax(), mask.data.shape)
    else:
        x, y = 0, 0

    if class_map is not None:
        caption = class_map.get_id(label)
    else:
        caption = str(label)

    return _draw_label(img=img, caption=caption, x=int(x), y=int(y), color=color)


def _draw_label(
    img: np.ndarray,
    caption: str,
    x: int,
    y: int,
    color,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale: float = 1.0,
):
    """ Draws a caption above the box in an image.
    """
    y -= 10
    w, h = cv2.getTextSize(caption, font, fontScale=font_scale, thickness=1)[0]

    # make the coords of the box with a small padding of two pixels
    box_pt1, box_pt2 = (x, y + 10), (x + w + 2, y - h - 2)
    cv2.rectangle(img, box_pt1, box_pt2, color, cv2.FILLED)

    cv2.putText(img, caption, (x, y), font, font_scale, (240, 240, 240), 2)

    return img


def draw_record(record, class_map: Optional[ClassMap] = None):
    sample = default_prepare_record(record)
    return draw_sample(sample=sample, class_map=class_map)


def draw_pred(
    img: np.ndarray, pred: dict,
):
    sample = pred.copy()
    sample["img"] = img
    return draw_sample(sample=sample)


def draw_bbox(
    img: np.ndarray, bbox: BBox, color: Tuple[int, int, int], thickness: int = 2
):
    """ Draws a box on an image with a given color.
    # Arguments
        image     : The image to draw on.
        box       : A list of 4 elements (x1, y1, x2, y2).
        color     : The color of the box.
        thickness : The thickness of the lines to draw a box with.
    """
    xyxy = tuple(np.array(bbox.xyxy, dtype=int))
    cv2.rectangle(img, xyxy[:2], xyxy[2:], color, thickness, cv2.LINE_AA)
    return img


def draw_mask(
    img: np.ndarray, mask: MaskArray, color: Tuple[int, int, int], blend: float = 0.5
):
    color = np.asarray(color, dtype=int)
    # draw mask
    mask_idxs = np.where(mask.data)
    img[mask_idxs] = blend * img[mask_idxs] + (1 - blend) * color

    # draw border
    border = mask.data - cv2.erode(mask.data, np.ones((7, 7), np.uint8), iterations=1)
    border_idxs = np.where(border)
    img[border_idxs] = color

    return img

