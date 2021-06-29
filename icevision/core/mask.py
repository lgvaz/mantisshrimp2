__all__ = [
    "Mask",
    "MaskArray",
    "MaskFile",
    "VocMaskFile",
    "RLE",
    "Polygon",
    "EncodedRLEs",
    "SemanticMaskFile",
]

from icevision.imports import *
from icevision.utils import *
from PIL import Image


class Mask(ABC):
    @abstractmethod
    def to_mask(self, h, w) -> "MaskArray":
        pass

    @abstractmethod
    def to_erles(self, h, w) -> "EncodedRLEs":
        pass


class EncodedRLEs(Mask):
    def __init__(self, erles: List[dict] = None):
        self.erles = erles or []

    def __repr__(self):
        return f"<{self.__class__.__name__} with {len(self)} objects>"

    def __len__(self):
        return len(self.erles)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.erles == other.erles
        return False

    def append(self, v: "EncodedRLEs"):
        self.erles.extend(v.erles)

    def extend(self, v: List["EncodedRLEs"]):
        for o in v:
            self.append(o)

    def pop(self, i: int):
        self.erles.pop(i)

    def to_mask(self, h, w) -> "MaskArray":
        mask = mask_utils.decode(self.erles)
        mask = mask.transpose(2, 0, 1)  # channels first
        return MaskArray(mask)

    def to_erles(self, h, w) -> "EncodedRLEs":
        return self


# TODO: Assert shape? (bs, height, width)
class MaskArray(Mask):
    """Binary numpy array representation of a mask.

    # Arguments
        data: Mask array, with the dimensions: (num_instances, height, width)
    """

    def __init__(self, data: np.uint8):
        self.data = data.astype(np.uint8)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return type(self)(self.data[i])

    def to_tensor(self):
        return tensor(self.data, dtype=torch.uint8)

    def to_mask(self, h, w):
        return self

    def to_erles(self, h, w) -> EncodedRLEs:
        return EncodedRLEs(
            mask_utils.encode(np.asfortranarray(self.data.transpose(1, 2, 0)))
        )

    def to_coco_rle(self, h, w) -> List[dict]:
        """From https://stackoverflow.com/a/49547872/6772672"""
        assert self.data.shape[1:] == (h, w)
        rles = []
        for mask in self.data:
            counts = []
            flat = itertools.groupby(mask.ravel(order="F"))
            for i, (value, elements) in enumerate(flat):
                if i == 0 and value == 1:
                    counts.append(0)
                counts.append(len(list(elements)))
            rles.append({"counts": counts, "size": (h, w)})
        return rles

    @property
    def shape(self):
        return self.data.shape

    @classmethod
    def from_masks(cls, masks: Union[EncodedRLEs, Sequence[Mask]], h: int, w: int):
        # HACK: check for backwards compatibility
        if isinstance(masks, EncodedRLEs):
            return masks.to_mask(h, w)
        else:
            masks_arrays = [o.to_mask(h=h, w=w).data for o in masks]
            return cls(np.concatenate(masks_arrays))


class MaskFile(Mask):
    """Holds the path to mask image file.

    # Arguments
        filepath: Path to the mask image file.
    """

    def __init__(self, filepath: Union[str, Path]):
        self.filepath = Path(filepath)

    def to_mask(self, h, w):
        mask = open_img(self.filepath, gray=True)
        obj_ids = np.unique(mask)[1:]
        masks = mask == obj_ids[:, None, None]
        return MaskArray(masks)

    def to_coco_rle(self, h, w) -> List[dict]:
        return self.to_mask(h=h, w=w).to_coco_rle(h=h, w=w)

    def to_erles(self, h, w) -> EncodedRLEs:
        return self.to_mask(h, w).to_erles(h, w)


class VocMaskFile(MaskFile):
    """Extension of `MaskFile` for VOC masks.
    Removes the color pallete and optionally drops void pixels.

    # Arguments
        drop_void (bool): drops the void pixels, which should have the value 255.
        filepath: Path to the mask image file.
    """

    def __init__(self, filepath: Union[str, Path], drop_void: bool = True):
        super().__init__(filepath=filepath)
        self.drop_void = drop_void

    def to_mask(self, h, w) -> MaskArray:
        mask_arr = np.array(Image.open(self.filepath))
        obj_ids = np.unique(mask_arr)[1:]
        masks = mask_arr == obj_ids[:, None, None]

        if self.drop_void:
            masks = masks[:-1, ...]

        return MaskArray(masks)


class RLE(Mask):
    """Run length encoding of a mask.

    Don't instantiate this class directly, instead use the classmethods
    `from_coco` and `from_kaggle`.
    """

    def __init__(self, counts: List[int]):
        self.counts = counts

    def to_mask(self, h, w) -> "MaskArray":
        return self.to_erles(h=h, w=w).to_mask(h=h, w=w)
        # Convert kaggle counts to mask
        # "From https://www.kaggle.com/julienbeaulieu/imaterialist-detectron2"
        # mask = np.full(h * w, 0, dtype=np.uint8)
        # for start, ones in zip(self.counts[::2], self.counts[1::2]):
        #     # counting starts on one
        #     start -= 1
        #     if ones:
        #         mask[start : start + ones] = 1
        # mask = mask.reshape((h, w), order="F")
        # return MaskArray(mask)

    def to_coco(self) -> List[int]:
        return self.counts

    def to_erles(self, h, w) -> EncodedRLEs:
        return EncodedRLEs(
            mask_utils.frPyObjects([{"counts": self.to_coco(), "size": [h, w]}], h, w)
        )

    @classmethod
    def from_string(cls, s, sep=" "):
        return cls(lmap(int, s.split(sep)))

    @classmethod
    def from_kaggle(cls, counts: Sequence[int]):
        """Described [here](https://www.kaggle.com/c/imaterialist-fashion-2020-fgvc7/overview/evaluation)"""
        if len(counts) % 2 != 0:
            raise ValueError("Counts must be divisible by 2")

        current = 1
        coco_counts = []
        for start, count in zip(counts[::2], counts[1::2]):
            coco_counts.append(start - current)  # zeros
            coco_counts.append(count)  # ones
            current = start + count

        # remove trailing zero
        if coco_counts[-1] == 0:
            coco_counts.pop(-1)

        return cls.from_coco(coco_counts)

    @classmethod
    def from_coco(cls, counts: Sequence[int]):
        """Described [here](https://stackoverflow.com/a/49547872/6772672)"""
        return cls(counts)
        # Convert from kaggle to coco
        # when counts is odd, round it with 0 ones at the end
        # if len(counts) % 2 != 0:
        #     counts = counts + [0]
        #
        # kaggle_counts, total = [], 0
        # for zeros, ones in zip(counts[::2], counts[1::2]):
        #     start = zeros + total + 1
        #     kaggle_counts.extend([start, ones])
        #     total += zeros + ones
        # return cls.from_kaggle(kaggle_counts)


class Polygon(Mask):
    """Polygon representation of a mask

    # Arguments
        points: The vertices of the polygon in the COCO standard format.
    """

    def __init__(self, points: List[List[int]]):
        self.points = points

    def to_mask(self, h, w):
        return self.to_erles(h=h, w=w).to_mask(h=h, w=w)

    def to_erles(self, h, w) -> EncodedRLEs:
        erles = mask_utils.frPyObjects(self.points, h, w)
        erle = mask_utils.merge(erles)  # make unconnected polygons a single mask
        return EncodedRLEs([erle])


class SemanticMaskFile(Mask):
    """Holds the path to mask image file.

    # Arguments
        filepath: Path to the mask image file.
    """

    def __init__(self, filepath: Union[str, Path], binary=False):
        self.filepath = Path(filepath)
        self.binary = binary

    def to_mask(self, h, w):
        # TODO: convert the 255 masks
        mask = open_img(self.filepath, gray=True)
        # HACK: because open_img now return PIL
        mask = np.array(mask)

        # convert 255 pixels to 1
        if self.binary:
            mask[mask == 255] = 1

        return MaskArray(mask[None])

    def to_coco_rle(self, h, w) -> List[dict]:
        raise NotImplementedError
        return self.to_mask(h=h, w=w).to_coco_rle(h=h, w=w)

    def to_erles(self, h, w) -> EncodedRLEs:
        # HACK: Doesn't make sense to convert to ERLE?
        return self
