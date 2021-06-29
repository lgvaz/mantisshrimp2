__all__ = ["RecordCollection"]

from icevision.imports import *
from icevision.utils import *
from icevision.core.record import BaseRecord, autofix_records
from icevision.data.data_splitter import DataSplitter


class RecordCollection:
    def __init__(self, create_record_fn):
        self.create_record_fn = create_record_fn
        self._records = IndexableDict()

    def get_by_record_id(self, record_id):
        try:
            record = self._records[record_id]
            record.is_new = False
            return record
        except KeyError:
            record = self._records[record_id] = self.create_record_fn()
            record.set_record_id(record_id)
            record.is_new = True
            return record

    def new(self, records: Sequence[BaseRecord]):
        new = type(self)(self.create_record_fn)
        new._records = IndexableDict([(record.record_id, record) for record in records])
        return new

    def make_splits(self, data_splitter: DataSplitter):
        record_id_splits = data_splitter.split(self)
        return [
            self.new([self._records[record_id] for record_id in record_ids])
            for record_ids in record_id_splits
        ]

        # for record_ids in record_id_splits:
        #     yield self.new([self._records[record_id] for record_id in record_ids])

    def autofix(self, show_pbar: int = True):
        records = autofix_records(self._records.values())
        return self.new(records)

    def __getitem__(self, i: int) -> BaseRecord:
        return self._records.values()[i]

    def __len__(self):
        return len(self._records)
