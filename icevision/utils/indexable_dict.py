__all__ = ["IndexableDict"]

import collections


class IndexableDictValuesView(collections.abc.ValuesView):
    def __getitem__(self, index):
        return self._mapping._list[index]


class IndexableDict(collections.UserDict):
    def __init__(self, *args, **kwargs):
        self._list = []
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self._list.append(value)

    def __delitem__(self, key):
        super().__delitem__(key)
        self._list.remove(key)

    def values(self) -> IndexableDictValuesView:
        return IndexableDictValuesView(self)
