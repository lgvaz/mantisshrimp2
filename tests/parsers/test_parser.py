import pytest
from icevision.all import *


@pytest.fixture
def data():
    return [
        {"id": 1, "filepath": __file__, "labels": ["a"], "bboxes": [[1, 2, 3, 4]]},
        {
            "id": 42,
            "filepath": __file__,
            "labels": ["a", "b"],
            "bboxes": [[1, 2, 3, 4], [10, 20, 30, 40]],
        },
        {"id": 3, "filepath": "none.txt", "labels": ["a"], "bboxes": [[1, 2, 3, 4]]},
    ]


class SimpleParser(parsers.Parser):
    def __init__(self, data):
        self.data = data
        super().__init__(
            template_record=BaseRecord(
                (
                    FilepathRecordComponent(),
                    InstancesLabelsRecordComponent(),
                    BBoxesRecordComponent(),
                )
            )
        )
        self.class_map = ClassMap(["a", "b"])

    def __iter__(self):
        yield from self.data

    def record_id(self, o) -> Hashable:
        return o["id"]

    def labels(self, o):
        return o["labels"]

    def parse_fields(self, o, record, is_new):
        record.set_filepath(o["filepath"])
        record.set_img_size(ImgSize(100, 100))

        record.detection.set_class_map(self.class_map)
        record.detection.add_labels(self.labels(o))
        record.detection.add_bboxes([BBox.from_xyxy(*pnts) for pnts in o["bboxes"]])


def test_parser(data, tmpdir):
    parser = SimpleParser(data)

    cache_filepath = Path(tmpdir / "simple_parser.pkl")
    records = parser.parse(data_splitter=SingleSplitSplitter())[0]
    assert cache_filepath.exists() == False
    assert len(records) == 2

    record = records[1]
    # assert set(record.keys()) == {
    #     "class_map",
    #     "record_id",
    #     "filepath",
    #     "height",
    #     "width",
    #     "labels",
    #     "bboxes",
    # }
    assert record.record_id == 42
    assert record.filepath == Path(__file__)
    assert len(record.detection.class_map) == 3
    assert record.detection.class_map.get_by_name("background") == 0
    assert record.detection.class_map.get_by_id(0) == "background"
    assert record.detection.label_ids == [1, 2]
    assert record.detection.bboxes == [
        BBox.from_xyxy(1, 2, 3, 4),
        BBox.from_xyxy(10, 20, 30, 40),
    ]

    assert parser._check_path() == False
    assert parser._check_path(cache_filepath) == False
    records = parser.parse(
        data_splitter=SingleSplitSplitter(), cache_filepath=cache_filepath
    )[0]
    assert parser._check_path(cache_filepath) == True
    assert cache_filepath.exists() == True
    loaded_records = pickle.load(open(cache_filepath, "rb"))[0]
    assert len(loaded_records) == len(records)
    for loaded_record, record in zip(loaded_records, records):
        assert loaded_record.filepath == record.filepath
        assert loaded_record.detection.label_ids == record.detection.label_ids
        assert loaded_record.detection.bboxes == record.detection.bboxes

    parser = SimpleParser(data)
    assert len(parser.class_map) == 3
    assert parser.class_map._lock == True


@pytest.mark.skip
def test_parser_annotation_len_mismatch(data):
    class BrokenParser(SimpleParser):
        def labels(self, o) -> List[Hashable]:
            return o["labels"][:1]

    parser = BrokenParser(data)

    with pytest.raises(RuntimeError) as err:
        records = parser.parse(data_splitter=SingleSplitSplitter())[0]
    assert "inconsistent number of annotations" in str(err)


@pytest.mark.skip
@pytest.mark.parametrize(
    "bbox_xyxy", [[1, 2, 1, 4], [3, 2, 1, 4], [1, 2, 3, 2], [1, 4, 3, 2]]
)
def test_parser_invalid_data(data, bbox_xyxy):
    invalid_sample = [{"id": 3, "labels": [1], "bboxes": [bbox_xyxy]}]

    parser = SimpleParser(data + invalid_sample)

    with pytest.raises(InvalidDataError) as err:
        records = parser.parse(data_splitter=SingleSplitSplitter())[0]
