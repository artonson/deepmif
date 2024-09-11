from abc import ABC, abstractmethod
from typing import List, Tuple, Union

from deepmif.dataset.dataset_entry import DatasetEntry, DatasetEntryOrNone


class Filter(ABC):
    @abstractmethod
    def __call__(self, data_entry: DatasetEntry) -> DatasetEntry:
        pass


FilterTupleOrList = Union[List[Filter], Tuple[Filter]]


class FilterList(Filter):
    def __init__(self, filters: FilterTupleOrList) -> None:
        self.filters = filters

    def __call__(self, data_entry: DatasetEntry) -> DatasetEntryOrNone:
        for filter in self.filters:
            data_entry = filter(data_entry)
        return data_entry


FilterListOrNone = Union[FilterList, None]
