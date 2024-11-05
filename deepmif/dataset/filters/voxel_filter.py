import numpy as np
import open3d as o3d

from deepmif.dataset.dataset_entry import DatasetEntry
from deepmif.dataset.filters.filter import Filter


class VoxelFilter(Filter):
    def __init__(self, voxel_size: float, is_centered=True):
        self.voxel_size = voxel_size
        self.is_centered = is_centered

    def __call__(self, data_entry: DatasetEntry) -> DatasetEntry:
        points = data_entry.point_cloud

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(
            points - np.tile(data_entry.pose[:3, 3].T, [len(points), 1])
            if not self.is_centered
            else points
        )

        pcd, trace, _ = pcd.voxel_down_sample_and_trace(
            self.voxel_size,
            pcd.get_min_bound(),
            pcd.get_max_bound(),
            approximate_class=True,
        )

        mask = np.max(trace, 1)

        points_filtered = points[mask]

        filtered_entry = DatasetEntry(
            data_entry.index, data_entry.pose, points_filtered
        )

        return filtered_entry
