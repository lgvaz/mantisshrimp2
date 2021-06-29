import pytest
from icevision.all import *
import albumentations as A
from icevision.models.inference import postprocess_bbox
from icevision.models.ross import efficientdet


def test_e2e_detect(samples_source, fridge_efficientdet_model, fridge_class_map):
    img_path = samples_source / "fridge/odFridgeObjects/images/10.jpg"
    tfms_ = tfms.A.Adapter([*tfms.A.resize_and_pad(384), tfms.A.Normalize()])
    bboxes = efficientdet.end2end_detect(
        img_path, tfms_, fridge_efficientdet_model, fridge_class_map
    )
    assert len(bboxes) == 2


def test_inference_postprocess_bbox(samples_source, fridge_efficientdet_model):
    img_path = samples_source / "fridge/odFridgeObjects/images/10.jpg"
    img = PIL.Image.open(img_path)
    w, h = img.size

    tfms_ = tfms.A.Adapter([*tfms.A.resize_and_pad(384), tfms.A.Normalize()])
    infer_ds = Dataset.from_images([np.array(img)], tfms_)
    pred = efficientdet.predict(fridge_efficientdet_model, infer_ds)[0]

    for bbox in pred.pred.detection.bboxes:
        xmin, ymin, xmax, ymax = postprocess_bbox(
            img, bbox, tfms_.tfms_list, pred.pred.height, pred.pred.width
        )
        assert xmin > 0
        assert xmax < w
        assert ymin > 0
        assert ymax < h


def _test_preds(preds):
    assert len(preds) == 1

    pred = preds[0].pred
    assert len(pred.detection.scores) == 2

    np.testing.assert_equal(pred.detection.label_ids, [2, 1])

    assert isinstance(pred.detection.bboxes[0], BBox)
    bboxes_np = np.array([bbox.xyxy for bbox in pred.detection.bboxes])
    bboxes_expected = np.array([[65, 59, 170, 261], [123, 212, 341, 292]])
    np.testing.assert_allclose(bboxes_np, bboxes_expected, atol=1)


def test_efficient_det_predict(fridge_efficientdet_model, fridge_efficientdet_records):
    fridge_efficientdet_model.eval()

    ds = fridge_efficientdet_records
    preds = efficientdet.predict(model=fridge_efficientdet_model, dataset=ds)

    _test_preds(preds)


def test_efficient_det_predict_from_dl(
    fridge_efficientdet_model, fridge_efficientdet_records
):
    fridge_efficientdet_model.eval()

    infer_dl = efficientdet.infer_dl(fridge_efficientdet_records, batch_size=1)
    preds = efficientdet.predict_from_dl(
        model=fridge_efficientdet_model, infer_dl=infer_dl, show_pbar=False
    )

    _test_preds(preds)


def test_efficient_det_predict_from_dl_threshold(
    fridge_efficientdet_model, fridge_efficientdet_records
):
    fridge_efficientdet_model.eval()

    infer_dl = efficientdet.infer_dl(fridge_efficientdet_records, batch_size=1)
    preds = efficientdet.predict_from_dl(
        model=fridge_efficientdet_model,
        infer_dl=infer_dl,
        show_pbar=False,
        detection_threshold=1.0,
    )

    assert len(preds[0].pred.detection.label_ids) == 0
