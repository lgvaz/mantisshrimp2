import pytest
from icevision.all import *


def test_bbox_parser(via_dir, via_bbox_class_map):
    parser = parsers.via(via_dir / "via_bbox.json", via_dir, via_bbox_class_map)
    records = parser.parse(data_splitter=SingleSplitSplitter())[0]
    assert len(records) == 2

    record = records[0]
    assert record.record_id == "IMG_4908.jpg"
    assert record.filepath == via_dir / "IMG_4908.jpg"
    assert record.width == 640
    assert record.height == 480

    assert record.detection.label_ids == [4, 1, 3, 2]
    # Polygon
    assert record.detection.bboxes[0].xyxy == (104, 120, 155, 176)
    # Rect
    assert record.detection.bboxes[3].xyxy == (263, 55, 308, 103)


def test_bbox_parser_alt_label(via_dir, via_bbox_class_map):
    parser = parsers.via(
        via_dir / "via_bbox.json", via_dir, via_bbox_class_map, label_field="object"
    )
    records = parser.parse(data_splitter=SingleSplitSplitter())[0]
    record = records[0]
    # Additional label is found when using the 'object' label_field.
    # When 'label' is used, one region is marked '###' and therefore not found in class_map.
    assert record.detection.label_ids == [4, 1, 3, 3, 2]


def test_bbox_parser_broken_label(via_dir, via_bbox_class_map):
    parser = parsers.via(
        via_dir / "via_bbox.json", via_dir, via_bbox_class_map, label_field="qualities"
    )
    with pytest.raises(parsers.VIAParseError, match=r"Non-string.*IMG_4908.*"):
        _ = parser.parse(data_splitter=SingleSplitSplitter())[0]


def test_bbox_parser_missing_label(via_dir, via_bbox_class_map):
    parser = parsers.via(
        via_dir / "via_bbox.json", via_dir, via_bbox_class_map, label_field="MISSING"
    )
    with pytest.raises(parsers.VIAParseError, match=r"Could not find.*IMG_4908.*"):
        _ = parser.parse(data_splitter=SingleSplitSplitter())[0]
