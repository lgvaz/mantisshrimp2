import pytest
from icevision.all import *


def _test_dl(x, y, recs):
    assert len(recs) == 1
    assert recs[0].img is None
    assert x.shape == torch.Size([1, 3, 64, 64])

    assert recs[0].segmentation.class_map is not None
    assert recs[0].segmentation.class_map.num_classes == 32

    if y is not None:
        assert y.shape == torch.Size([1, 64, 64])

        # check maximum value in y is not higher than 31, given we have 32 classes in total
        assert y.max() < 32
        assert y.min() >= 0


def test_train_dataloader(camvid_ds):
    train_ds, _ = camvid_ds
    dl = models.fastai.unet.train_dl(
        train_ds, batch_size=1, num_workers=0, shuffle=False
    )

    (x, y), records = first(dl)
    assert records[0].record_id == "0006R0_f02340"

    _test_dl(x, y, records)


def test_valid_dataloader(camvid_ds):
    _, valid_ds = camvid_ds
    dl = models.fastai.unet.valid_dl(
        valid_ds, batch_size=1, num_workers=0, shuffle=False
    )

    (x, y), records = first(dl)
    assert records[0].record_id == "0006R0_f02400"

    _test_dl(x, y, records)


def test_infer_dataloader(camvid_ds):
    _, valid_ds = camvid_ds
    dl = models.fastai.unet.infer_dl(
        valid_ds, batch_size=1, num_workers=0, shuffle=False
    )
    (x,), records = first(dl)
    assert records[0].record_id == "0006R0_f02400"

    _test_dl(x, None, records)
