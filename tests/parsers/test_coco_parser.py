import pytest
from icevision.all import *


@pytest.fixture
def coco_bbox_parser(coco_dir):
    return parsers.coco(coco_dir / "annotations.json", coco_dir / "images", mask=False)


@pytest.fixture
def coco_mask_parser(coco_dir):
    return parsers.coco(coco_dir / "annotations.json", coco_dir / "images", mask=True)


def test_keypoints_parser(coco_dir, coco_keypoints_parser):
    records = coco_keypoints_parser.parse(data_splitter=SingleSplitSplitter())[0]
    assert len(records) == 2
    record = records[1]
    assert record.filepath == coco_dir / "images/000000404249.jpg"
    assert len(record.detection.keypoints) == 1
    assert record.detection.keypoints[0].n_visible_keypoints == 16
    assert record.detection.keypoints[0].y.max() == 485

    assert len(records[0].detection.keypoints) == 3


def test_bbox_parser(coco_dir, coco_bbox_parser):
    records = coco_bbox_parser.parse(data_splitter=SingleSplitSplitter())[0]
    assert len(records) == 5

    record = records[0]
    assert record.record_id == 343934
    assert record.filepath == coco_dir / "images/000000343934.jpg"
    assert record.width == 640
    assert record.height == 480

    assert len(record.detection.class_map) == 91
    assert record.detection.class_map.get_by_id(90) == "toothbrush"
    assert record.detection.label_ids == [4]
    assert pytest.approx(record.detection.bboxes[0].xyxy) == (
        175.14,
        175.68,
        496.2199,
        415.68,
    )
    assert record.detection.iscrowds == [0]
    assert pytest.approx(record.detection.areas) == [43522.805]


def test_mask_parser(coco_mask_parser):
    records = coco_mask_parser.parse(data_splitter=SingleSplitSplitter())[0]
    assert len(records) == 5

    record = records[0]
    assert len(record.detection.masks) == 1
    assert record.detection.masks[0].points == [
        [
            457.3,
            258.92,
            458.38,
            276.22,
            467.03,
            289.19,
            473.51,
            305.41,
            483.24,
            334.59,
            492.97,
            359.46,
            496.22,
            390.81,
            492.97,
            415.68,
            478.92,
            397.3,
            472.43,
            382.16,
            464.86,
            361.62,
            452.97,
            340,
            437.84,
            334.59,
            428.11,
            337.84,
            415.14,
            354.05,
            387.03,
            373.51,
            362.16,
            361.62,
            339.46,
            351.89,
            311.35,
            348.65,
            307.03,
            352.97,
            301.62,
            363.78,
            295.14,
            373.51,
            283.24,
            392.97,
            274.59,
            402.7,
            231.35,
            409.19,
            210.81,
            403.78,
            190.27,
            383.24,
            182.7,
            367.03,
            185.95,
            336.76,
            197.84,
            312.97,
            205.41,
            306.49,
            229.19,
            289.19,
            254.05,
            280.54,
            256.22,
            258.92,
            234.59,
            242.7,
            220.54,
            247.03,
            208.65,
            258.92,
            202.16,
            271.89,
            180.54,
            285.95,
            177.3,
            287.03,
            175.14,
            266.49,
            178.38,
            250.27,
            176.22,
            236.22,
            175.14,
            222.16,
            188.11,
            209.19,
            212.97,
            212.43,
            234.59,
            220,
            256.22,
            223.24,
            297.3,
            227.57,
            318.92,
            234.05,
            330.81,
            235.14,
            351.35,
            226.49,
            360,
            218.92,
            376.22,
            216.76,
            398.92,
            217.84,
            418.38,
            213.51,
            420.54,
            201.62,
            419.46,
            196.22,
            435.68,
            183.24,
            449.73,
            175.68,
            468.11,
            176.76,
            464.86,
            192.97,
            460.54,
            196.22,
            458.38,
            207.03,
            464.86,
            221.08,
            468.11,
            234.05,
            469.19,
            240.54,
            468.11,
            243.78,
        ]
    ]
