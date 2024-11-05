from dataclasses import dataclass
from typing import Union

from deepmif.dataset.types import PointCloudXx3, Transform4x4


@dataclass
class DatasetEntry:
    index: int
    pose: Transform4x4
    point_cloud: PointCloudXx3


DatasetEntryOrNone = Union[DatasetEntry, None]
