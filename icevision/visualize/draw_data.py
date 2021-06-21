# The draw functions are based from:
# https://github.com/fizyr/keras-retinanet/blob/master/keras_retinanet/utils/visualization.py
# https://github.com/fizyr/keras-maskrcnn/blob/master/keras_maskrcnn/utils/visualization.py

__all__ = [
    "draw_sample",
    "draw_record",
    "draw_pred",
    "draw_bbox",
    "draw_mask",
    "draw_keypoints",
]

from icevision.imports import *
from icevision.data import *
from icevision.core import *
from icevision.visualize.utils import *

# This should probably move elsewhere
from PIL import Image, ImageFont, ImageDraw
import PIL

DEFAULT_FONT_PATH = get_default_font()


def draw_sample(
    sample,
    class_map: Optional[ClassMap] = None,
    denormalize_fn: Optional[callable] = None,
    display_label: bool = True,
    display_bbox: bool = True,
    display_score: bool = True,
    display_mask: bool = True,
    display_keypoints: bool = True,
    font_path: Optional[os.PathLike] = DEFAULT_FONT_PATH,
    font_size: Union[int, float] = None,
    label_color: Union[np.array, list, tuple, str] = "#C4C4C4",  # Mild Gray
    label_border_color: Union[np.array, list, tuple, str] = "#020303",  # Black,
    label_thin_border: bool = True,
    label_pad_width_factor: float = 0.02,
    label_pad_height_factor: float = 0.005,
    mask_blend: float = 0.5,
    mask_border_thickness: int = 7,
    color_map: Optional[dict] = None,  # label -> color mapping
    prettify: bool = True,
    prettify_func: Callable = str.capitalize,
    return_as_pil_img=False,
    # Args for plotting specific labels
    exclude_labels: List[str] = [],
    include_only: List[str] = None,
    multiple_classification_spacing_factor: float = 1.05,
    dynamic_font_size_div_factor: float = 20.0,
    include_classification_task_names: bool = True,
    include_instances_task_names: bool = False,
) -> Union[np.ndarray, PIL.Image.Image]:
    """
    Selected kwargs:

    * label_color: A <collection> of RGB values or a hex code string that defines
                   the color of all the plotted labels
    * label_border_color: Color of the border around the label
    * label_thin_border: Apply a thin border around the label. If false, applies
                         a thick border. If None, applies no border
    * label_pad_width_factor: Amount of padding to apply relative to the image's width.
                              Applies padding to bbox coords if padding bbox
                              labels else to the top-left of the image for classif labels
    * mask_blend: Degree of transparency of the mask. 1 = opaque, 0 = transparent
    * mask_border_thickness: Degree of thickness of the mask. Must be an odd number
    * color_map: An optional dictionary that maps the label => color-value
    * prettify: Format labels based on `prettify_func`
    * prettify_func: A string -> string processing function
    * return_as_pil_image: If true, returns the sample as a PIL image, else np.array
    * exclude_labels: (Optional) List of labels that you'd like to exclude from being plotted
    * include_only: (Optional) List of labels that must be exclusively plotted. Takes
                    precedence over `exclude_labels` (?)
    """
    img = np.asarray(sample.img).copy()  # HACK
    num_classification_plotted = 0

    # Dynamic font size based on image height
    if font_size is None:
        font_size = sample.img_size.height / dynamic_font_size_div_factor

    if denormalize_fn is not None:
        img = denormalize_fn(img)

    for task, composite in sample.task_composites.items():
        # Should break if no ClassMap found in composite.
        #  Should be as the only composite without ClassMap should be
        #  `sample.common`. This is a foundational assumption? #NOTE
        class_map = getattr(composite, "class_map", None)

        if composite.get_component_by_type(ClassificationLabelsRecordComponent):
            x = 0
            y = (
                font_size
                * num_classification_plotted
                * multiple_classification_spacing_factor
            )
            num_classification_plotted += 1
        else:
            x, y = None, None

        # HACK
        if hasattr(composite, "masks"):
            masks = composite.masks.to_mask(h=sample.height, w=sample.width)
        else:
            masks = []

        for label, bbox, mask, keypoints, score in itertools.zip_longest(
            getattr(composite, "labels", []),  # list of strings
            getattr(composite, "bboxes", []),
            masks,
            getattr(composite, "keypoints", []),
            getattr(composite, "scores", []),
        ):
            # random color by default
            color = (np.random.random(3) * 0.6 + 0.4) * 255

            # logic for plotting specific labels only
            # `include_only` > `exclude_labels`
            if not label == []:
                # label_str = (
                #     class_map.get_by_name(label) if class_map is not None else ""
                # )
                if include_only is not None:
                    if not label in include_only:
                        continue
                elif label in exclude_labels:
                    continue

            # if color-map is given and `labels` are predicted
            # then set color accordingly
            if color_map is not None:
                color = as_rgb_tuple(color_map[label])
                color = np.array(color).astype(np.float)

            if display_mask and mask is not None:
                img = draw_mask(
                    img=img,
                    mask=mask,
                    color=color,
                    blend=mask_blend,
                    border_thickness=mask_border_thickness,
                )
            if display_bbox and bbox is not None:
                img = draw_bbox(img=img, bbox=bbox, color=color)
            if display_keypoints and keypoints is not None:
                img = draw_keypoints(img=img, kps=keypoints, color=color)
            if display_label and label is not None:
                prefix = ""
                if include_classification_task_names:
                    if composite.get_component_by_type(
                        ClassificationLabelsRecordComponent
                    ):
                        prefix = prettify_func(task) + ": "
                if include_instances_task_names:
                    if composite.get_component_by_type(InstancesLabelsRecordComponent):
                        prefix = prettify_func(task) + ": "

                img = draw_label(
                    img=img,
                    label=label,
                    score=score if display_score else None,
                    bbox=bbox,
                    mask=mask,
                    class_map=class_map,
                    color=label_color,
                    border_color=label_border_color,
                    pad_width_factor=label_pad_width_factor,
                    pad_height_factor=label_pad_height_factor,
                    thin_border=label_thin_border,
                    font_size=font_size,
                    font=font_path,
                    prettify=prettify,
                    prettify_func=prettify_func,
                    return_as_pil_img=False,  # should this always be False??
                    prefix=prefix,
                    x=x,
                    y=y,
                )
    if return_as_pil_img:
        # may or may not be a PIL Image based on `display_label`
        return img if isinstance(img, PIL.Image.Image) else PIL.Image.fromarray(img)
    else:
        # will be a `np.ndarray` by default so no need for casting
        return img


def draw_label(
    img: np.ndarray,
    label: Union[int, str],
    score: Optional[float],
    color: Union[np.ndarray, list, tuple],
    border_color: Union[np.ndarray, list, tuple],
    class_map: Optional[ClassMap] = None,
    bbox=None,
    mask=None,
    font: Union[int, os.PathLike, None] = None,
    font_size: Union[int, float] = 12,
    prettify: bool = True,
    prettify_func: Callable = str.capitalize,
    return_as_pil_img=False,
    pad_width_factor=0.02,
    pad_height_factor=0.005,
    thin_border=True,
    x: Optional[int] = None,
    y: Optional[int] = None,
    prefix: str = "",
) -> Union[np.ndarray, PIL.Image.Image]:
    # finds label position based on bbox or mask
    if x is None or y is None:
        # print(f"X: {x}, Y: {y}")
        if bbox is not None:
            x, y, _, _ = bbox.xyxy
        elif mask is not None:
            y, x = np.unravel_index(mask.data.argmax(), mask.data.shape)
        else:
            x, y = 0, 0

    if class_map is not None:
        if isinstance(label, int):
            # TODO: This may never get triggered because we're looping
            # over composite.labels which is a list of strings
            caption = class_map.get_by_id(label)
        else:
            caption = label
    else:
        caption = str(label)
    if prettify:
        # We could introduce a callback here for more complex label renaming
        caption = prefix + caption
        caption = prettify_func(caption)

    # Append label confidence to caption if applicable
    if score is not None:
        if prettify:
            score = f"{score * 100: .2f}%"
        caption = f"{caption}: {score}"

    if not Path(font).exists():
        # PIL throws cryptic errors for wrong filepaths, so let's catch it earlier here
        raise FileNotFoundError(f"{font} file doesn't exist")
    return _draw_label(
        img=img,
        caption=caption,
        x=x,
        y=y,
        color=color,
        border_color=border_color,
        font_path=font,
        font_size=int(font_size),
        return_as_pil_img=return_as_pil_img,
        pad_width_factor=pad_width_factor,
        pad_height_factor=pad_height_factor,
        thin_border=thin_border,
    )


def _draw_label(
    img: np.ndarray,
    caption: str,
    x: int,
    y: int,
    color: Union[np.ndarray, list, tuple],
    border_color: Union[np.ndarray, list, tuple],
    font_path=DEFAULT_FONT_PATH,
    font_size: int = 20,
    return_as_pil_img: bool = False,
    pad_width_factor=0.02,
    pad_height_factor=0.005,
    thin_border=True,
) -> Union[PIL.Image.Image, np.ndarray]:
    """Draw labels on the image"""
    font = PIL.ImageFont.truetype(font_path, size=font_size)
    color = as_rgb_tuple(color)
    border_color = as_rgb_tuple(border_color)

    height, width = img.shape[:2]
    x_pad = height * pad_width_factor
    y_pad = width * pad_height_factor
    x, y = x + x_pad, y + y_pad

    img = PIL.Image.fromarray(img)
    draw = ImageDraw.Draw(img)

    if thin_border is not None:
        # Draw thin / thick border around text
        draw.text(
            (x - 1, y if thin_border else y - 1), caption, font=font, fill=border_color
        )
        draw.text(
            (x + 1, y if thin_border else y - 1), caption, font=font, fill=border_color
        )
        draw.text(
            (x if thin_border else x - 1, y - 1), caption, font=font, fill=border_color
        )
        draw.text(
            (x if thin_border else x + 1, y + 1), caption, font=font, fill=border_color
        )

    # Now draw text over the border
    draw.text((x, y), caption, font=font, fill=color)
    if return_as_pil_img:
        return img
    else:
        return np.array(img)


def draw_record(
    record,
    class_map: Optional[ClassMap] = None,
    display_label: bool = True,
    display_bbox: bool = True,
    display_mask: bool = True,
    display_score: bool = True,
    display_keypoints: bool = True,
    font_path: Optional[os.PathLike] = DEFAULT_FONT_PATH,
    font_size: Union[int, float] = 12,
    label_color: Union[np.array, list, tuple, str] = "#C4C4C4",  # Mild Gray
    mask_blend: float = 0.5,
    mask_border_thickness: int = 7,
    color_map: Optional[dict] = None,  # label -> color mapping
    prettify: bool = True,
    prettify_func: Callable = str.capitalize,
    return_as_pil_img=False,
    # Args for plotting specific labels
    exclude_labels: List[str] = [],
    include_only: List[str] = None,
):
    sample = record.load()
    return draw_sample(
        sample=sample,
        class_map=class_map,
        display_label=display_label,
        display_score=display_score,
        display_bbox=display_bbox,
        display_mask=display_mask,
        display_keypoints=display_keypoints,
        font_path=font_path,
        font_size=font_size,
        label_color=label_color,
        mask_blend=mask_blend,
        mask_border_thickness=mask_border_thickness,
        color_map=color_map,
        prettify=prettify,
        prettify_func=prettify_func,
        return_as_pil_img=return_as_pil_img,
        exclude_labels=exclude_labels,
        include_only=include_only,
    )


def draw_pred(
    pred: Prediction,
    denormalize_fn: Optional[callable] = None,
    display_label: bool = True,
    display_score: bool = True,
    display_bbox: bool = True,
    display_mask: bool = True,
    font_path: Optional[os.PathLike] = DEFAULT_FONT_PATH,
    font_size: Union[int, float] = 12,
    label_color: Union[np.array, list, tuple, str] = "#C4C4C4",  # Mild Gray
    mask_blend: float = 0.5,
    mask_border_thickness: int = 7,
    color_map: Optional[dict] = None,  # label -> color mapping
    prettify: bool = True,
    prettify_func: Callable = str.capitalize,
    return_as_pil_img=False,
    # Args for plotting specific labels
    exclude_labels: List[str] = [],
    include_only: List[str] = None,
):
    return draw_sample(
        sample=pred.pred,
        denormalize_fn=denormalize_fn,
        display_label=display_label,
        display_score=display_score,
        display_bbox=display_bbox,
        display_mask=display_mask,
        font_path=font_path,
        font_size=font_size,
        label_color=label_color,
        mask_blend=mask_blend,
        mask_border_thickness=mask_border_thickness,
        color_map=color_map,
        prettify=prettify,
        prettify_func=prettify_func,
        return_as_pil_img=return_as_pil_img,
        exclude_labels=exclude_labels,
        include_only=include_only,
    )


def draw_bbox(
    img: np.ndarray,
    bbox: BBox,
    color: Tuple[int, int, int],
    gap: bool = True,
):
    """Draws a box on an image with a given color.
    # Arguments
        image     : The image to draw on.
        box       : A list of 4 elements (x1, y1, x2, y2).
        color     : The color of the box.
    """

    # Calculate image dimensions
    dims = sorted(img.shape, reverse=True)
    color = as_rgb_tuple(color)
    img = PIL.Image.fromarray(img.squeeze()).convert("RGB")
    draw = PIL.ImageDraw.Draw(img)

    # corner thickness is linearly correlated with the smaller image dimension.
    # We use the smaller image dimension rather than image area so as to avoid
    # overly thick lines for large non-square images prior to transforming
    # images. We set lower and upper bounds for corner thickness.
    min_corner = 1
    max_corner = 15
    corner_thickness = int(0.005 * dims[1] + min_corner)
    if corner_thickness > max_corner:
        corner_thickness = int(max_corner)

    corner_length = int(0.021 * dims[1] + 2.25)

    # inner thickness of bboxes with corners
    inner_thickness = int(1 + 0.0005 * dims[1])

    # bbox thickness of bboxes without corners
    min_bbox = 1
    max_bbox = 8
    bbox_thickness = int(0.0041 * dims[1] - 0.0058)
    if bbox_thickness < min_bbox:
        bbox_thickness = min_bbox
    if bbox_thickness > max_bbox:
        bbox_thickness = int(max_bbox)

    if gap == False:
        xyxy = tuple(np.array(bbox.xyxy, dtype=int))
        draw.rectangle(xyxy, fill=None, outline=color, width=bbox_thickness)
        return np.array(img)

    xmin, ymin, xmax, ymax = tuple(np.array(bbox.xyxy, dtype=int))

    points = [0] * 12
    points[0] = (xmin, ymin + corner_length)
    points[1] = (xmin, ymin)
    points[2] = (xmin + corner_length, ymin)

    points[3] = (xmax - corner_length, ymin)
    points[4] = (xmax, ymin)
    points[5] = (xmax, ymin + corner_length)

    points[6] = (xmax, ymax - corner_length)
    points[7] = (xmax, ymax)
    points[8] = (xmax - corner_length, ymax)

    points[9] = (xmin + corner_length, ymax)
    points[10] = (xmin, ymax)
    points[11] = (xmin, ymax - corner_length)

    if (
        ymax - (ymin + 4 * corner_length) < corner_length
        or xmax - (xmin + 4 * corner_length) < corner_length
    ):
        for i in range(4):
            draw.line(
                xy=(points[i * 3 + 1], points[10 - 3 * i]),
                fill=color,
                width=bbox_thickness,
            )
        for i in range(2):
            draw.line(
                xy=(points[6 * i + 1], points[i * 6 + 4]),
                fill=color,
                width=bbox_thickness,
            )
    else:
        for i in range(2):
            for j in range(2):
                draw.line(
                    xy=(points[i * 6 + j * 4], points[i * 6 + j * 4 + 1]),
                    fill=color,
                    width=corner_thickness,
                    joint=None,
                )
                draw.line(
                    xy=(points[i * 3 + 1], points[10 - 3 * i]),
                    fill=color,
                    width=inner_thickness,
                    joint=None,
                )
        for i in range(2):
            for j in range(2):
                draw.line(
                    xy=(points[i * 6 + j * 2 + 1], points[i * 6 + j * 2 + 2]),
                    fill=color,
                    width=corner_thickness,
                    joint=None,
                )
                draw.line(
                    xy=(points[6 * i + 1], points[i * 6 + 4]),
                    fill=color,
                    width=inner_thickness,
                    joint=None,
                )

    return np.array(img)


def draw_mask(
    img: np.ndarray,
    mask: MaskArray,
    color: Tuple[int, int, int],
    blend: float = 0.5,
    border_thickness: int = 7,
):
    # Border thickness must be an odd integer
    if border_thickness % 2 == 0:
        # TODO: Shall we throw an error, or change the value to the nearest
        # even integer automatically and raise a warning?
        raise ValueError(
            f"`border_thickness` must be an odd number. You entered {border_thickness}"
        )
    img = PIL.Image.fromarray(img.squeeze())
    w, h = img.size

    mask_idxs = np.where(mask.data)

    # Add alpha with 0 transparency. We draw the mask with border color first
    color = np.append(color, 255)
    mask_arr = np.zeros((h, w, 4), dtype=np.uint8)
    mask_arr[mask_idxs] = color

    # Now create a second mask and draw the desired color on top of the
    # border mask. If `border_thickness` is 0, this replaces the border mask
    _mask = Image.fromarray(mask_arr)
    _mask = _mask.filter(PIL.ImageFilter.MinFilter(border_thickness))
    _mask_idx = np.where(_mask.convert("L", palette=Image.ADAPTIVE))
    mask_arr[_mask_idx] = np.append(color[:3], blend * 255)

    # Create RGBA PIL mask image
    mask_pil = PIL.Image.fromarray(mask_arr, mode="RGBA")

    # Blend everything keeping the alpha
    # Key concept is that alpha for non-mask pixels are 0 (transparent)
    img.putalpha(255)
    img = PIL.Image.alpha_composite(img, mask_pil)

    return np.array(img)


def draw_keypoints(
    img: np.ndarray,
    kps: KeyPoints,
    color: Tuple[int, int, int],
):
    x, y, v = kps.x, kps.y, kps.visible

    # calculate scaling for points and connections
    img_h, img_w, _ = img.shape
    img_area = img_h * img_w
    img = PIL.Image.fromarray(img.squeeze())
    draw = PIL.ImageDraw.Draw(img)
    dynamic_size = int(0.01867599 * (img_area ** 0.4422045))
    dynamic_size = max(dynamic_size, 1)

    # draw connections
    if kps.metadata is not None and kps.metadata.connections is not None:
        for connection in kps.metadata.connections:
            if v[connection.p1] > 0 and v[connection.p2] > 0:
                draw.line(
                    xy=(
                        (int(x[connection.p1]), int(y[connection.p1])),
                        (int(x[connection.p2]), int(y[connection.p2])),
                    ),
                    fill=connection.color,
                    width=dynamic_size,
                )

    # draw points
    for x_c, y_c in zip(x[v > 0], y[v > 0]):
        radius = dynamic_size
        x1 = x_c - radius
        x2 = x_c + radius
        y1 = y_c - radius
        y2 = y_c + radius
        draw.ellipse(
            xy=[x1, y1, x2, y2],
            # xy=[x1, x0, y1, y0],
            fill=as_rgb_tuple(color),
            outline=None,
            width=0,
        )

    return np.array(img)
