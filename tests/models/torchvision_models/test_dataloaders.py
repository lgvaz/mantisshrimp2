import pytest
from icevision.all import *
from icevision.models.torchvision import faster_rcnn, mask_rcnn, keypoint_rcnn


@pytest.fixture
def faster_rcnn_records(object_detection_record):
    return [object_detection_record.load()] * 2


@pytest.fixture
def mask_rcnn_records(instance_segmentation_record):
    return [instance_segmentation_record.load()] * 2


# TODO: Duplication with fasterrcnn records
# @pytest.fixture()
# def mask_records(img, labels, bboxes, masks):
#     record = InstanceSegmentationRecord()

#     record.set_record_id(1)
#     record.set_img(img)
#     record.add_labels(labels)
#     record.add_bboxes(bboxes)
#     record.add_masks([masks])
#     sample = record.load()
#     return [sample] * 2


### Faster RCNN ###
def _test_common_rcnn_batch(batch):
    (images, targets), records = batch

    assert isinstance(images, list)
    assert len(images) == 2
    assert images[0].dtype == torch.float
    assert images[0].shape == (3, 375, 500)

    assert isinstance(targets, list)
    assert isinstance(targets[0], dict)
    assert len(targets) == 2

    for target in targets:
        assert target["labels"].dtype == torch.int64
        assert torch.all(target["labels"] == tensor([1, 2], dtype=torch.int64))

        assert target["boxes"].dtype == torch.float
        assert torch.all(
            target["boxes"]
            == tensor([[1, 2, 3, 4], [10, 20, 30, 40]], dtype=torch.float)
        )


def _test_faster_rcnn_batch(batch):
    _test_common_rcnn_batch(batch)

    (images, targets), records = batch
    for target in targets:
        assert set(target) == {"labels", "boxes"}


def _test_infer_batch(batch):
    assert len(batch) == 1

    batch = batch[0]
    assert len(batch) == 2

    for x in batch:
        assert x.shape == (3, 375, 500)
        assert x.dtype == torch.float


def test_faster_rcnn_build_train_batch(faster_rcnn_records):
    batch = faster_rcnn.build_train_batch(faster_rcnn_records)
    _test_faster_rcnn_batch(batch=batch)


def test_faster_rcnn_build_train_batch_empty(empty_annotations_record):
    (_, targets), _ = faster_rcnn.build_train_batch([empty_annotations_record])

    for x in targets:
        assert x["labels"].equal(torch.zeros(0, dtype=torch.int64))
        assert x["boxes"].equal(torch.zeros((0, 4), dtype=torch.float32))


def test_faster_rcnn_build_valid_batch(faster_rcnn_records):
    batch = faster_rcnn.build_valid_batch(faster_rcnn_records)
    _test_faster_rcnn_batch(batch=batch)


def test_faster_rcnn_build_infer_batch(infer_dataset):
    batch, samples = faster_rcnn.build_infer_batch(infer_dataset)
    _test_infer_batch(batch)


def test_faster_rcnn_train_dataloader(faster_rcnn_records):
    dl = faster_rcnn.train_dl(faster_rcnn_records, batch_size=2)
    batch = first(dl)
    _test_faster_rcnn_batch(batch=batch)


def test_faster_rcnn_valid_dataloader(faster_rcnn_records):
    dl = faster_rcnn.valid_dl(faster_rcnn_records, batch_size=2)
    batch = first(dl)
    _test_faster_rcnn_batch(batch=batch)


def test_faster_rcnn_infer_dataloader(infer_dataset):
    dl = faster_rcnn.infer_dl(dataset=infer_dataset, batch_size=2)
    batch, samples = first(dl)
    _test_infer_batch(batch=batch)


### Mask RCNN ###
def _test_mask_rcnn_batch(batch):
    _test_common_rcnn_batch(batch=batch)
    (images, targets), records = batch

    for target in targets:
        assert set(target.keys()) == {"labels", "boxes", "masks"}
        assert target["masks"].shape == (2, 4, 4)
        assert target["masks"].dtype == torch.uint8


def test_mask_rcnn_build_train_batch(mask_rcnn_records):
    batch = mask_rcnn.build_train_batch(mask_rcnn_records)
    _test_mask_rcnn_batch(batch)


def test_mask_rcnn_build_train_batch_empty(empty_annotations_record):
    (_, targets), _ = mask_rcnn.build_train_batch([empty_annotations_record])

    height, width = empty_annotations_record.img.shape[:-1]
    for x in targets:
        assert x["labels"].equal(torch.zeros(0, dtype=torch.int64))
        assert x["boxes"].equal(torch.zeros((0, 4), dtype=torch.float32))
        assert x["masks"].equal(torch.zeros((0, height, width), dtype=torch.uint8))


def test_mask_rcnn_build_valid_batch(mask_rcnn_records):
    batch = mask_rcnn.build_valid_batch(mask_rcnn_records)
    _test_mask_rcnn_batch(batch)


def test_mask_rcnn_build_infer_batch(infer_dataset):
    test_faster_rcnn_infer_dataloader(infer_dataset)


def test_mask_rcnn_train_dataloader(mask_rcnn_records):
    dl = mask_rcnn.train_dl(mask_rcnn_records, batch_size=2)
    batch = first(dl)
    _test_mask_rcnn_batch(batch=batch)


def test_mask_rcnn_valid_dataloader(mask_rcnn_records):
    dl = mask_rcnn.valid_dl(mask_rcnn_records, batch_size=2)
    batch = first(dl)
    _test_mask_rcnn_batch(batch=batch)


def test_mask_rcnn_infer_dataloader(infer_dataset):
    test_faster_rcnn_infer_dataloader(infer_dataset)


def test_keypoints_rcnn_dataloader(coco_keypoints_parser):
    import random

    random.seed(40)
    records = coco_keypoints_parser.parse(data_splitter=SingleSplitSplitter())[0]
    tfm = tfms.A.Adapter(
        [*tfms.A.aug_tfms(size=384, presize=512, crop_fn=None), tfms.A.Normalize()]
    )
    train_ds = Dataset(records, tfm)
    train_dl = keypoint_rcnn.train_dl(
        train_ds, batch_size=2, num_workers=0, shuffle=True
    )
    (x, y), recs = first(train_dl)

    assert len(x) == len(y) == 2
    assert x[0].shape == x[1].shape == torch.Size([3, 384, 384])

    ind = [r.filepath.parts[-1] for r in recs].index("000000128372.jpg")
    assert y[ind]["keypoints"].shape == torch.Size([3, 17, 3])
    assert y[ind]["labels"].tolist() == [1, 1, 1]
    assert y[-(ind - 1)]["keypoints"].shape == torch.Size([1, 17, 3])
    assert y[-(ind - 1)]["boxes"].shape == torch.Size([1, 4])
