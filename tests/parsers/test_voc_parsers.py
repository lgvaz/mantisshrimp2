from icevision.all import *


def test_voc_annotation_parser(
    samples_source, voc_class_map, check_attributes_on_component
):
    annotation_parser = parsers.VOCBBoxParser(
        annotations_dir=samples_source / "voc/Annotations",
        images_dir=samples_source / "voc/JPEGImages",
        class_map=voc_class_map,
    )
    records = annotation_parser.parse(data_splitter=SingleSplitSplitter())[0]

    assert len(records) == 2

    record = records[0]
    assert record.detection.class_map == voc_class_map
    assert record.record_id == "2007_000063"
    assert record.filepath == samples_source / "voc/JPEGImages/2007_000063.jpg"
    assert record.width == 500
    assert record.height == 375
    assert record.detection.label_ids == [
        voc_class_map.get_by_name(k) for k in ["dog", "chair"]
    ]
    assert record.detection.bboxes == [
        BBox.from_xyxy(123, 115, 379, 275),
        BBox.from_xyxy(75, 1, 428, 375),
    ]

    record = records[1]
    assert record.detection.class_map == voc_class_map
    assert record.record_id == "2011_003353"
    assert record.filepath == samples_source / "voc/JPEGImages/2011_003353.jpg"
    assert record.height == 500
    assert record.width == 375
    assert record.detection.label_ids == [voc_class_map.get_by_name("person")]
    assert record.detection.bboxes == [BBox.from_xyxy(130, 45, 375, 470)]

    check_attributes_on_component(records[0])
    check_attributes_on_component(records[1])


def test_voc_mask_parser(samples_source, voc_class_map):
    parser = parsers.VOCMaskParser(
        annotations_dir=samples_source / "voc/Annotations",
        images_dir=samples_source / "voc/JPEGImages",
        class_map=voc_class_map,
        masks_dir=samples_source / "voc/SegmentationClass",
    )

    records = parser.parse(data_splitter=SingleSplitSplitter())[0]

    assert len(records) == 1

    record = records[0]
    assert record.detection.class_map == voc_class_map
    assert record.record_id == "2007_000063"
    assert record.filepath == samples_source / "voc/JPEGImages/2007_000063.jpg"
    assert record.height == 375
    assert record.width == 500
    assert record.detection.label_ids == [
        voc_class_map.get_by_name(k) for k in ["dog", "chair"]
    ]
    assert record.detection.bboxes == [
        BBox.from_xyxy(123, 115, 379, 275),
        BBox.from_xyxy(75, 1, 428, 375),
    ]
    masks = record.detection.masks
    assert len(masks) == 1
    assert masks[0].filepath == samples_source / "voc/SegmentationClass/2007_000063.png"
    # assert masks[0].filepath == Path(
    #     "/home/lgvaz/git/icevision/samples/voc/SegmentationClass/2007_000063.png"
    # )
