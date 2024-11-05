import numpy as np

from deepmif.dataset.dataset_entry import DatasetEntry
from deepmif.dataset.filters.filter import Filter


class ApplyPose(Filter):
    def __init__(self, invert=False):
        self.invert = invert

    def __call__(self, data_entry: DatasetEntry) -> DatasetEntry:
        points = data_entry.point_cloud
        pose = np.linalg.inv(data_entry.pose) if self.invert else data_entry.pose

        points = np.column_stack((points, np.ones(len(points))))
        points = np.einsum("jk,ik->ij", pose, points)[:, :3]

        filtered_entry = DatasetEntry(data_entry.index, data_entry.pose, points)

        return filtered_entry
