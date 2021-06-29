__all__ = [
    "draw_label",
    "bbox_polygon",
    "draw_mask",
    "as_rgb_tuple",
    "get_default_font",
    "rand_cmap",
]

from icevision.imports import *
from icevision.utils import *
from matplotlib import patches
from PIL import Image, ImageFont, ImageDraw
import PIL


def draw_label(ax, x, y, name, color, fontsize=18):
    ax.text(
        x + 1,
        y - 2,
        name,
        fontsize=fontsize,
        color="white",
        va="bottom",
        bbox=dict(facecolor=color, edgecolor=color, pad=2, alpha=0.9),
    )


def bbox_polygon(bbox):
    bx, by, bw, bh = bbox.xywh
    poly = np.array([[bx, by], [bx, by + bh], [bx + bw, by + bh], [bx + bw, by]])
    return patches.Polygon(poly)


def draw_mask(ax, mask, color):
    color_mask = np.ones((*mask.shape, 3)) * color
    ax.imshow(np.dstack((color_mask, mask * 0.5)))
    ax.contour(mask, colors=[color_mask[0, 0, :]], alpha=0.4)


def as_rgb_tuple(x: Union[np.ndarray, tuple, list, str]) -> tuple:
    "Convert np RGB values -> tuple for PIL compatibility"
    if isinstance(x, (np.ndarray, tuple, list)):
        if not len(x) == 3:
            raise ValueError(f"Expected 3 (RGB) numbers, got {len(x)}")
        if isinstance(x, np.ndarray):
            return tuple(x.astype(np.int))
        elif isinstance(x, tuple):
            return x
        elif isinstance(x, list):
            return tuple(x)
    elif isinstance(x, str):
        return PIL.ImageColor.getrgb(x)
    else:
        raise ValueError(f"Expected {{np.ndarray|list|tuple}}, got {type(x)}")


def get_default_font() -> str:
    import requests

    font_dir = get_root_dir() / "fonts"
    font_dir.mkdir(exist_ok=True)

    font_file = font_dir / "SpaceGrotesk-Medium.ttf"
    if not font_file.exists():
        URL = "https://raw.githubusercontent.com/airctic/storage/master/SpaceGrotesk-Medium.ttf"
        logger.info(
            "Downloading default `.ttf` font file - SpaceGrotesk-Medium.ttf from {} to {}",
            URL,
            font_file,
        )
        font_file.write_bytes(requests.get(URL).content)
    return str(font_file)


# Generate random colormap from https://stackoverflow.com/questions/14720331/how-to-generate-random-colors-in-matplotlib/32520273#32520273
def rand_cmap(
    nlabels,
    type="bright",
    first_color_black=True,
    last_color_black=False,
    verbose=False,
):
    """
    Creates a random colormap to be used together with matplotlib. Useful for segmentation tasks
    :param nlabels: Number of labels (size of colormap)
    :param type: 'bright' for strong colors, 'soft' for pastel colors
    :param first_color_black: Option to use first color as black, True or False
    :param last_color_black: Option to use last color as black, True or False
    :param verbose: Prints the number of labels and shows the colormap. True or False
    :return: colormap for matplotlib
    """
    from matplotlib.colors import LinearSegmentedColormap
    import colorsys

    np.random.seed(0)

    if type not in ("bright", "soft"):
        print('Please choose "bright" or "soft" for type')
        return

    if verbose:
        print("Number of labels: " + str(nlabels))

    # Generate color map for bright colors, based on hsv
    if type == "bright":
        randHSVcolors = [
            (
                np.random.uniform(low=0.0, high=1),
                np.random.uniform(low=0.2, high=1),
                np.random.uniform(low=0.9, high=1),
            )
            for i in range(nlabels)
        ]

        # Convert HSV list to RGB
        randRGBcolors = []
        for HSVcolor in randHSVcolors:
            randRGBcolors.append(
                colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2])
            )

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]

        random_colormap = LinearSegmentedColormap.from_list(
            "new_map", randRGBcolors, N=nlabels
        )

    # Generate soft pastel colors, by limiting the RGB spectrum
    if type == "soft":
        low = 0.6
        high = 0.95
        randRGBcolors = [
            (
                np.random.uniform(low=low, high=high),
                np.random.uniform(low=low, high=high),
                np.random.uniform(low=low, high=high),
            )
            for i in range(nlabels)
        ]

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]
        random_colormap = LinearSegmentedColormap.from_list(
            "new_map", randRGBcolors, N=nlabels
        )

    # Display colorbar
    if verbose:
        from matplotlib import colors, colorbar
        from matplotlib import pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(15, 0.5))

        bounds = np.linspace(0, nlabels, nlabels + 1)
        norm = colors.BoundaryNorm(bounds, nlabels)

        cb = colorbar.ColorbarBase(
            ax,
            cmap=random_colormap,
            norm=norm,
            spacing="proportional",
            ticks=None,
            boundaries=bounds,
            format="%1i",
            orientation="horizontal",
        )

    return random_colormap
