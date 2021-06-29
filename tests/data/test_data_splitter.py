import pytest
from icevision.all import *


@pytest.fixture
def records():
    def create_record_func():
        return BaseRecord([])

    records = RecordCollection(create_record_func)
    for record_id in ["file1", "file2", "file3", "file4"]:
        records.get_by_record_id(record_id)
    return records


def test_single_split_splitter(records):
    data_splitter = SingleSplitSplitter()
    splits = data_splitter(records)
    assert splits == [["file1", "file2", "file3", "file4"]]


def test_random_splitter(records):
    data_splitter = RandomSplitter([0.6, 0.2, 0.2], seed=42)
    splits = data_splitter(records)
    np.testing.assert_equal(splits, [["file2", "file4"], ["file1"], ["file3"]])


def test_fixed_splitter(records):
    presplits = [["file4", "file3"], ["file2"], ["file1"]]

    data_splitter = FixedSplitter(presplits)
    splits = data_splitter(records)
    assert splits == presplits
