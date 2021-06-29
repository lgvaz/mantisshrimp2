__all__ = [
    "DataSplitter",
    "SingleSplitSplitter",
    "RandomSplitter",
    "FixedSplitter",
]

from icevision.imports import *
from icevision.utils import *
from icevision.core import *


class DataSplitter(ABC):
    """Base class for all data splitters."""

    def __call__(self, records: Sequence[BaseRecord]):
        return self.split(records=records)

    @abstractmethod
    def split(self, records: Sequence[BaseRecord]):
        """Splits `ids` into groups.

        # Arguments
            idmap: idmap used for getting ids.
        """
        pass


class SingleSplitSplitter(DataSplitter):
    """Return all items in a single group, without shuffling."""

    def split(self, records: Sequence[BaseRecord]):
        """Puts all `ids` in a single group.

        # Arguments
            idmap: idmap used for getting ids.
        """
        return [[record.record_id for record in records]]


class RandomSplitter(DataSplitter):
    """Randomly splits items.

    # Arguments
        probs: `Sequence` of probabilities that must sum to one. The length of the
            `Sequence` is the number of groups to to split the items into.
        seed: Internal seed used for shuffling the items. Define this if you need
            reproducible results.

    # Examples

        Split data into three random groups.

        ```python
        idmap = IDMap(["file1", "file2", "file3", "file4"])

        data_splitter = RandomSplitter([0.6, 0.2, 0.2], seed=42)
        splits = data_splitter(idmap)

        np.testing.assert_equal(splits, [[1, 3], [0], [2]])
        ```
    """

    def __init__(self, probs: Sequence[int], seed: int = None):
        self.probs = probs
        self.seed = seed

    def split(self, records: Sequence[BaseRecord]):
        """Randomly splits `ids` based on parameters passed to the constructor of this class.

        # Arguments
            idmap: idmap used for getting ids.
        """
        # calculate split indexes
        p = np.array(self.probs) * len(records)  # convert percentage to absolute
        p = np.ceil(p).astype(int)  # round up, so each split has at least one example
        p[p.argmax()] -= sum(p) - len(
            records
        )  # removes excess from split with most items
        p = np.cumsum(p)

        with np_local_seed(self.seed):
            shuffled = np.random.permutation([record.record_id for record in records])

        return np.split(shuffled, p.tolist())[:-1]  # last element is always empty


class FixedSplitter(DataSplitter):
    """Split `ids` based on predefined splits.

    # Arguments:
        splits: The predefined splits.

    # Examples

        Split data into three pre-defined groups.
        ```python
        idmap = IDMap(["file1", "file2", "file3", "file4"])
        presplits = [["file4", "file3"], ["file2"], ["file1"]]

        data_splitter = FixedSplitter(presplits)
        splits = data_splitter(idmap=idmap)

        assert splits == [[3, 2], [1], [0]]
        ```
    """

    def __init__(self, splits: Sequence[Sequence[Hashable]]):
        self.splits = splits

    def split(self, records: Sequence[BaseRecord]):
        """Execute the split

        # Arguments
            idmap: idmap used for getting ids.
        """
        return self.splits
