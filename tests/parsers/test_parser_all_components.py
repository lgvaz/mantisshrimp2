import pytest
from icevision.all import *


# TODO: This seems to be an uncessary test, maybe not, it tests parse_fields without
# all the loop around it
@pytest.fixture
def dummy_parser_all_components():
    class MyParser(Parser):
        def __init__(self):
            record = BaseRecord(
                (
                    FilepathRecordComponent(),
                    InstancesLabelsRecordComponent(),
                    BBoxesRecordComponent(),
                    InstanceMasksRecordComponent(),
                    KeyPointsRecordComponent(),
                    AreasRecordComponent(),
                    IsCrowdsRecordComponent(),
                )
            )
            super().__init__(template_record=record)

        def __iter__(self) -> Any:
            raise NotImplementedError

        def parse_fields(self, o, record: BaseRecord, is_new) -> None:
            record.set_filepath(__file__)
            record.set_img_size(ImgSize(480, 420))

            record.detection.set_class_map(ClassMap(["a"]))
            record.detection.add_labels(["a"])
            record.detection.add_bboxes([BBox.from_xyxy(1, 2, 3, 4)])
            record.detection.add_masks([MaskArray(np.zeros((1, 420, 480)))])
            record.detection.add_keypoints([KeyPoints((1, 1, 1), None)])
            record.detection.add_areas([4.2])
            record.detection.add_iscrowds([False])

    return MyParser


def test_parser_parse_fields(dummy_parser_all_components):
    parser = dummy_parser_all_components()

    record = parser.create_record()
    parser.parse_fields(None, record, is_new=True)

    assert record.filepath == Path(__file__)
    assert record.height == 420
    assert record.width == 480

    assert record.detection.label_ids == [1]
    assert record.detection.bboxes == [BBox.from_xyxy(1, 2, 3, 4)]
    assert np.array_equal(record.detection.masks[0].data, np.zeros((1, 420, 480)))
    assert record.detection.areas == [4.2]
    assert record.detection.iscrowds == [False]
    assert record.detection.keypoints == [KeyPoints((1, 1, 1), None)]
