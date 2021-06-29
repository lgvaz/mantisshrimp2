import pytest
from icevision.all import *
from icevision.models.ross import efficientdet


@pytest.fixture
def expected_coco_metric_output():
    return [
        " Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.494",
        " Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.556",
        " Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.556",
        " Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000",
        " Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000",
        " Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.507",
        " Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.450",
        " Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.850",
        " Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.850",
        " Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000",
        " Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000",
        " Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.850",
    ]


@pytest.fixture()
def expected_confusion_matrix_output():
    return [
        "[[0 0 0 0 0]",
        " [0 0 0 0 0]",
        " [0 0 1 0 0]",
        " [0 1 0 0 0]",
        " [0 0 0 0 0]]",
    ]


@pytest.mark.parametrize(
    "metric, expected_output",
    [
        (SimpleConfusionMatrix(print_summary=True), "expected_confusion_matrix_output"),
        (COCOMetric(print_summary=True), "expected_coco_metric_output"),
    ],
)
def test_efficientdet_metrics(
    fridge_efficientdet_model,
    fridge_efficientdet_records,
    metric,
    expected_output,
    request,
):
    expected_output = request.getfixturevalue(expected_output)
    fridge_efficientdet_model.eval()

    batch, records = efficientdet.build_valid_batch(fridge_efficientdet_records)

    raw_preds = fridge_efficientdet_model(*batch)

    preds = efficientdet.convert_raw_predictions(
        batch=batch,
        raw_preds=raw_preds["detections"],
        records=fridge_efficientdet_records,
        detection_threshold=0.0,
    )

    metric.accumulate(preds)

    with CaptureStdout() as output:
        metric.finalize()

    assert output == expected_output


def test_plot_confusion_matrix(fridge_efficientdet_model, fridge_efficientdet_records):
    fridge_efficientdet_model.eval()

    batch, records = efficientdet.build_valid_batch(fridge_efficientdet_records)

    raw_preds = fridge_efficientdet_model(*batch)

    preds = efficientdet.convert_raw_predictions(
        batch=batch,
        raw_preds=raw_preds["detections"],
        records=fridge_efficientdet_records,
        detection_threshold=0.0,
    )

    cm = SimpleConfusionMatrix()
    cm.accumulate(preds)
    cm.finalize()
    cm.plot()
