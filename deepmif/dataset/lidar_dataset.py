import copy
from dataclasses import dataclass

import open3d as o3d
import torch
from numpy.linalg import inv
from torch.utils.data import Dataset

from deepmif.dataset.data_sampler import (
    LidarDataSampler,
    LidarSamplingConfig,
    LidarSamplingInput,
)
from deepmif.dataset.dataset_entry import DatasetEntry
from deepmif.model.feature_octree import FeatureOctree
from deepmif.model.mif_decoder import MIFDecoderInput
from deepmif.utils import is_none


@dataclass
class LiDARDatasetPool:
    begin_pose_inv = None

    points = None
    points_init = None
    sensor_origins = None
    sdf = None

    def __len__(self):
        return len(self.points)

    def __getitem__(self, index):
        return (
            self.points[index],
            self.points_init[index],
            self.sensor_origins[index],
            self.sdf[index],
        )


class LiDARDataset(Dataset):
    def __init__(
        self,
        sampler_config,
        scale,
        batch_size,
        octree: FeatureOctree = None,
        dtype=torch.float32,
        device="cuda",
        pool_device="cpu",
    ) -> None:
        super().__init__()

        self.scale = scale
        self.batch_size = batch_size
        self.octree = octree
        self.dtype = dtype
        self.device = device
        self.pool_device = pool_device

        torch.set_default_dtype(self.dtype)

        # initialize the data sampler
        lidar_config = LidarSamplingConfig(**sampler_config)
        self.lidar_sampler = LidarDataSampler(lidar_config, self.device)

        # merged downsampled point cloud
        self.map_down_pc = o3d.geometry.PointCloud()
        # map bounding box in the world coordinate system
        self.map_bbx = o3d.geometry.AxisAlignedBoundingBox()
        self.cur_bbx = o3d.geometry.AxisAlignedBoundingBox()

        self.pools = LiDARDatasetPool()

    def to_tensor(self, arr, device=None):
        _device = self.pool_device if device is None else device
        return torch.tensor(arr, dtype=self.dtype, device=_device)

    def init_pool_or_append(
        self, pool: LiDARDatasetPool, arr: torch.Tensor, device=None
    ):
        _device = self.pool_device if device is None else device
        _pool = self.pools.__getattribute__(pool)
        if is_none(_pool):
            self.pools.__setattr__(pool, arr.to(_device))
        else:
            self.pools.__setattr__(
                pool,
                torch.cat((_pool, arr.to(_device)), dim=0),
            )

    def process_frame(self, d_entry: DatasetEntry):
        if is_none(self.pools.begin_pose_inv):
            self.pools.begin_pose_inv = inv(d_entry.pose)

        # map visualization
        frame_pc_clone = o3d.geometry.PointCloud()
        frame_pc_clone.points = o3d.utility.Vector3dVector(d_entry.point_cloud)

        self.map_down_pc += frame_pc_clone
        self.cur_frame_pc = frame_pc_clone

        self.map_bbx = self.map_down_pc.get_axis_aligned_bounding_box()
        self.cur_bbx = self.cur_frame_pc.get_axis_aligned_bounding_box()

        sampling_input = LidarSamplingInput(
            self.to_tensor(d_entry.point_cloud, device=self.device),
            self.to_tensor(d_entry.pose[:3, 3], device=self.device),
        )

        sampling_output = self.lidar_sampler(sampling_input)

        sensor_origins = sampling_input.sensor_origin.reshape((1, 3))
        sensor_origins = sensor_origins.repeat(len(sampling_output.sdf), 1)

        # update feature octree
        if self.octree is not None:
            surface_points = sampling_output.points[sampling_output.surface_mask]
            surface_points = surface_points * self.scale
            self.octree.update(surface_points, False)

        self.init_pool_or_append("points", sampling_output.points)
        self.init_pool_or_append("points_init", sampling_input.points)
        self.init_pool_or_append("sensor_origins", sensor_origins)
        self.init_pool_or_append("sdf", sampling_output.sdf)

    def write_merged_pc(self, out_path):
        map_down_pc_out = copy.deepcopy(self.map_down_pc)
        map_down_pc_out.transform(inv(self.pools.begin_pose_inv))
        o3d.io.write_point_cloud(out_path, map_down_pc_out)
        print("save the merged point cloud map to %s\n" % (out_path))

    def __len__(self) -> int:
        return len(self.pools)

    def get_batch(self):
        index = torch.randint(0, len(self.pools), (self.batch_size,)).to(
            self.pool_device
        )

        (points, points_init, sensor_origins, sdf) = [
            sample.to(self.device) for sample in self.pools[index]
        ]

        return MIFDecoderInput(
            points=points,
            sensor_origins=sensor_origins,
            points_init=points_init,
            sdf=sdf,
        )
