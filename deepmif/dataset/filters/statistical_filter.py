import open3d as o3d

from deepmif.dataset.dataset_entry import DatasetEntry
from deepmif.dataset.filters.filter import Filter


class StatisticalFilter(Filter):
    def __init__(self, nb_neighbors: int, std_ratio: float):
        self.nb_neighbors = nb_neighbors
        self.std_ratio = std_ratio

    def __call__(self, data_entry: DatasetEntry) -> DatasetEntry:
        points = data_entry.point_cloud

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        pcd, mask = pcd.remove_statistical_outlier(
            self.nb_neighbors,
            self.std_ratio,
        )

        points_filtered = points[mask]

        filtered_entry = DatasetEntry(
            data_entry.index, data_entry.pose, points_filtered
        )

        return filtered_entry
