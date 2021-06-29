from icevision.tfms.albumentations.albumentations_adapter import (
    AlbumentationsMasksComponent,
)
import pytest
from icevision.all import *
from albumentations.augmentations.transforms import (
    LongestMaxSize,
    Normalize,
    PadIfNeeded,
)
from icevision.tfms.albumentations.albumentations_helpers import (
    get_size_without_padding,
    get_transform,
)

# TODO: Check that attributes are being set on components
@pytest.fixture
def records(coco_mask_records):
    return coco_mask_records


def test_set_on_components(records, check_attributes_on_component):
    tfm = tfms.A.Adapter([tfms.A.HorizontalFlip(p=1.0)])
    tfm_ds = Dataset(records, tfm=tfm)

    tfmed = tfm_ds[0]
    check_attributes_on_component(tfmed)


def test_inference_transform(records, check_attributes_on_component):
    img = open_img(records[0].filepath)
    tfm = tfms.A.Adapter([tfms.A.HorizontalFlip(p=1.0)])
    ds = Dataset.from_images([img], tfm)

    tfmed = ds[0]
    assert (tfmed.img == np.array(img)[:, ::-1, :]).all()
    check_attributes_on_component(tfmed)


def test_simple_transform(records, check_attributes_on_component):
    tfm = tfms.A.Adapter([tfms.A.HorizontalFlip(p=1.0)])
    ds = Dataset(records)
    tfm_ds = Dataset(records, tfm=tfm)

    sample, tfmed = ds[0], tfm_ds[0]
    assert (tfmed.img == sample.img[:, ::-1, :]).all()
    check_attributes_on_component(tfmed)


def test_crop_transform(records, check_attributes_on_component):
    tfm = tfms.A.Adapter([tfms.A.CenterCrop(100, 100, p=1.0)])
    tfm_ds = Dataset(records, tfm=tfm)
    tfmed = tfm_ds[0]
    # assert "keypoints" not in tfm_ds.tfm.tfms.processors.keys()
    assert len(tfmed.detection.label_ids) == 1
    assert len(tfmed.detection.bboxes) == 1
    assert len(tfmed.detection.masks) == 1
    assert len(tfmed.detection.iscrowds) == 1
    assert len(tfmed.detection.areas) == 1
    check_attributes_on_component(tfmed)


def test_crop_transform_empty(records, check_attributes_on_component):
    tfm = tfms.A.Adapter([tfms.A.Crop(0, 0, 100, 100, p=1.0)])
    tfm_ds = Dataset(records, tfm=tfm)
    tfmed = tfm_ds[0]
    # assert "keypoints" not in tfm_ds.tfm.tfms.processors.keys()
    assert len(tfmed.detection.label_ids) == 0
    assert len(tfmed.detection.bboxes) == 0
    assert not tfmed.detection.mask_array.data.any()
    assert len(tfmed.detection.iscrowds) == 0
    assert len(tfmed.detection.areas) == 0
    check_attributes_on_component(tfmed)

    # assert orignal record was not changed
    record = records[0]
    assert len(record.detection.label_ids) == 1
    assert len(record.detection.bboxes) == 1
    assert len(record.detection.masks) == 1
    assert len(record.detection.iscrowds) == 1
    assert len(record.detection.areas) == 1
    check_attributes_on_component(record)


def test_keypoints_transform(coco_keypoints_parser, check_attributes_on_component):
    records = coco_keypoints_parser.parse(data_splitter=SingleSplitSplitter())[0]
    tfm = tfms.A.Adapter([tfms.A.Resize(427 * 2, 640 * 2), tfms.A.Normalize()])
    # assert "keypoints" in tfm.tfms.processors.keys()
    # assert "bboxes" in tfm.tfms.processors.keys()

    ds = Dataset(records)
    tfm_ds = Dataset(records, tfm=tfm)

    record, tfmed = ds[0], tfm_ds[0]
    # assert "keypoints" in tfm.tfms.processors.keys()
    # assert "bboxes" in tfm.tfms.processors.keys()
    assert len(record.detection.keypoints) == 3
    assert len(tfmed.detection.keypoints) == 3
    assert set([c for c in tfmed.detection.keypoints[0].visible]) == {0.0, 1.0, 2.0}
    assert set([c for c in record.detection.keypoints[0].visible]) == {0, 1, 2}
    assert (tfmed.detection.keypoints[0].x == record.detection.keypoints[0].x * 2).all()
    assert (tfmed.detection.keypoints[0].y == record.detection.keypoints[0].y * 2).all()
    check_attributes_on_component(tfmed)


def test_keypoints_transform_crop_error(coco_keypoints_parser):
    records = coco_keypoints_parser.parse(data_splitter=SingleSplitSplitter())[0]
    tfm = tfms.A.Adapter([*tfms.A.aug_tfms(size=384, presize=512), tfms.A.Normalize()])

    tfm_ds = Dataset(records, tfm=tfm)
    with pytest.raises(RuntimeError):
        tfm_ds[0]


def test_filter_keypoints():
    tfms_kps, w, h, v = (
        [(0, 0), (60, 119), (-30, 40), (100, 300), (30, 100)],
        80,
        120,
        [0, 1, 1, 1, 2],
    )
    img_size = ImgSize(width=w, height=h)
    tra_n = tfms.A.AlbumentationsKeypointsComponent._remove_albu_outside_keypoints(
        tfms_kps, v, img_size
    )

    assert len(tfms_kps) == len(tra_n)
    assert tra_n == [(0, 0, 0), (60, 119, 1), (0, 0, 0), (0, 0, 0), (30, 100, 2)]

    tfms_kps, w, h, v = (
        [(0, 0), (79, 119), (-30, 40), (100, 300), (70, 100)],
        120,
        120,
        [0, 1, 1, 1, 2],
    )
    tra_n = tfms.A.AlbumentationsKeypointsComponent._remove_albu_outside_keypoints(
        tfms_kps, v, img_size
    )

    assert len(tfms_kps) == len(tra_n)
    assert tra_n == [(0, 0, 0), (79, 119, 1), (0, 0, 0), (0, 0, 0), (70, 100, 2)]


def test_filter_boxes():
    inp = (52.17503641656451, 274.5014178489639, 123.51860681160832, 350.33471836373275)
    out = (52.17503641656451, 274.5014178489639, 123.51860681160832, 320)
    h, w = 256, 384

    res = tfms.A.AlbumentationsBBoxesComponent._clip_bboxes(inp, h, w)
    assert out == res


def test_get_size_without_padding() -> None:
    """Test get_size_without_padding."""
    img = PIL.Image.fromarray(np.uint8(np.zeros((15, 15, 3))))
    transforms = [*tfms.A.resize_and_pad(20), tfms.A.Normalize()]
    assert get_size_without_padding(transforms, img, 20, 20) == (20, 20)

    img = PIL.Image.fromarray(np.uint8(np.zeros((20, 15, 3))))
    transforms = [*tfms.A.resize_and_pad(20), tfms.A.Normalize()]
    assert get_size_without_padding(transforms, img, 20, 20) == (20, 15)

    img = PIL.Image.fromarray(np.uint8(np.zeros((15, 10, 3))))
    transforms = [*tfms.A.resize_and_pad(20), tfms.A.Normalize()]
    assert get_size_without_padding(transforms, img, 20, 20) == (20, 13)


def test_get_transform() -> None:
    """Test get_transform."""
    transforms = [*tfms.A.resize_and_pad(20), tfms.A.Normalize()]

    assert get_transform(transforms, "not there") is None
    assert isinstance(get_transform(transforms, "Normalize"), Normalize)
    assert isinstance(get_transform(transforms, "Pad"), PadIfNeeded)
    assert isinstance(get_transform(transforms, "Longest"), LongestMaxSize)
