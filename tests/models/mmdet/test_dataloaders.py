import pytest
from icevision.all import *


@pytest.fixture()
def img():
    return 255 * np.ones((4, 4, 3), dtype=np.uint8)


@pytest.fixture()
def labels():
    return ["1", "2"]


@pytest.fixture()
def bboxes():
    return [BBox.from_xyxy(1, 2, 3, 4), BBox.from_xyxy(10, 20, 30, 40)]


@pytest.fixture()
def masks():
    return MaskArray(np.ones((2, 4, 4), dtype=np.uint8))


@pytest.fixture()
def class_map():
    return ClassMap(classes=["1", "2"])


@pytest.fixture()
def mask_records(img, labels, bboxes, masks, class_map):
    record = BaseRecord(
        (
            ImageRecordComponent(),
            InstancesLabelsRecordComponent(),
            BBoxesRecordComponent(),
            InstanceMasksRecordComponent(),
        )
    )
    record.set_record_id(1)
    record.set_img(img)
    record.detection.set_class_map(class_map)
    record.detection.add_labels(labels)
    record.detection.add_bboxes(bboxes)
    record.detection.add_masks([masks])
    sample = record.load()
    return [sample] * 2


def test_bbox_dataloader(fridge_ds, fridge_class_map):
    train_ds, _ = fridge_ds
    train_dl = models.mmdet.faster_rcnn.train_dl(
        train_ds, batch_size=2, num_workers=0, shuffle=False
    )
    data, recs = first(train_dl)

    assert set(data.keys()) == {"gt_bboxes", "gt_labels", "img", "img_metas"}
    assert (
        (np.array(recs[0].detection.label_ids) - 1) == data["gt_labels"][0].numpy()
    ).all()
    assert (
        (np.array(recs[1].detection.label_ids) - 1) == data["gt_labels"][1].numpy()
    ).all()
    assert data["img_metas"][0]["img_shape"] == (384, 384, 3)
    assert data["img_metas"][1]["pad_shape"] == (384, 384, 3)
    assert (
        data["img_metas"][1]["scale_factor"] == np.array([1.0, 1.0, 1.0, 1.0])
    ).all()
    assert recs[0].detection.class_map == fridge_class_map
    assert recs[1].detection.class_map == fridge_class_map


def _test_common_mmdet_mask_batch(batch):
    data, recs = batch

    assert set(data.keys()) == {
        "gt_bboxes",
        "gt_labels",
        "gt_masks",
        "img",
        "img_metas",
    }
    assert (
        (np.array(recs[0].detection.label_ids) - 1) == data["gt_labels"][0].numpy()
    ).all()
    assert data["img_metas"][0]["img_shape"] == (4, 4, 3)
    assert (
        data["img_metas"][0]["scale_factor"] == np.array([1.0, 1.0, 1.0, 1.0])
    ).all()
    assert "BitmapMasks" in str(type(data["gt_masks"][0]))
    assert len(data["gt_masks"]) == 2
    assert data["gt_masks"][0].height == data["gt_masks"][0].width == 4

    rec = recs[0]
    assert isinstance(rec.detection.label_ids, list)
    assert isinstance(rec.detection.bboxes[0], BBox)
    assert isinstance(rec.detection.mask_array, MaskArray)
    assert isinstance(rec.height, int)
    assert isinstance(rec.width, int)
    assert isinstance(rec.record_id, int)
    assert isinstance(rec.img, np.ndarray)


def test_mmdet_mask_rcnn_build_train_batch(mask_records):
    batch = models.mmdet.mask_rcnn.build_train_batch(mask_records)
    _test_common_mmdet_mask_batch(batch)


def test_mmdet_mask_rcnn_build_valid_batch(mask_records):
    batch = models.mmdet.mask_rcnn.build_valid_batch(mask_records)
    _test_common_mmdet_mask_batch(batch)


def test_mmdet_mask_rcnn_build_infer_batch(mask_records):
    batch = models.mmdet.mask_rcnn.build_infer_batch(mask_records)
    data, recs = batch

    assert set(data.keys()) == {
        "img",
        "img_metas",
    }
    assert recs[0].detection.label_ids == [1, 2]
    assert data["img_metas"][0][0]["img_shape"] == (4, 4, 3)
    assert (
        data["img_metas"][0][0]["scale_factor"] == np.array([1.0, 1.0, 1.0, 1.0])
    ).all()

    rec = recs[0]
    assert isinstance(rec.detection.label_ids, list)
    assert isinstance(rec.detection.bboxes[0], BBox)
    assert isinstance(rec.detection.mask_array, MaskArray)
    assert isinstance(rec.height, int)
    assert isinstance(rec.width, int)
    assert isinstance(rec.record_id, int)
    assert isinstance(rec.img, np.ndarray)
