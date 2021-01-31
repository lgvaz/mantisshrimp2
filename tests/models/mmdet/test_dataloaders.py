import pytest
from icevision.all import *


def test_faster_dataloader(fridge_ds, fridge_class_map):
    train_ds, _ = fridge_ds
    train_dl = models.mmdet.faster_rcnn.train_dl(
        train_ds, batch_size=2, num_workers=0, shuffle=False
    )
    data, recs = first(train_dl)

    assert set(data.keys()) == {"gt_bboxes", "gt_labels", "img", "img_metas"}
    assert ((np.array(recs[0]["labels"]) - 1) == data["gt_labels"][0].numpy()).all()
    assert ((np.array(recs[1]["labels"]) - 1) == data["gt_labels"][1].numpy()).all()
    assert data["img_metas"][0]["img_shape"] == (384, 384, 3)
    assert data["img_metas"][1]["pad_shape"] == (384, 384, 3)
    assert (
        data["img_metas"][1]["scale_factor"] == np.array([1.0, 1.0, 1.0, 1.0])
    ).all()
    assert recs[0]["class_map"] == fridge_class_map
    assert recs[1]["class_map"] == fridge_class_map
    assert recs[1]["img"].shape == data["img"][1].numpy().swapaxes(0, -1).shape
