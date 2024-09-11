from dataclasses import dataclass

import torch


@dataclass
class LidarSamplingConfig:
    around_surface_sampling_num: int = 3
    around_surface_sampling_dist: float = 0.3
    free_space_sampling_num: int = 3
    free_space_sampling_ratio: float = 0.7
    free_space_sampling_start_dist: float = 0.7
    occupied_space_sampling_num: int = 3
    occupied_space_sampling_dist: float = 0.3

    uniform_sampling: bool = False


@dataclass
class LidarSamplingInput:
    points: torch.Tensor
    sensor_origin: torch.Tensor


@dataclass
class LidarSamplingOutput:
    points: torch.Tensor
    sdf: torch.Tensor
    surface_mask: torch.Tensor


class LidarDataSampler:
    def __init__(self, config: LidarSamplingConfig, device):
        self.config = config
        self.device = device

    def get_samples_between(self, size, start, end):
        if self.config.uniform_sampling:
            num, smp = size
            samples = torch.linspace(0, 1, smp).reshape(1, smp).repeat((num, 1))
        else:
            samples = torch.rand(size, device=self.device)
        samples = samples.to(self.device)
        return samples * (end - start) + start

    def __call__(self, data: LidarSamplingInput) -> LidarSamplingOutput:
        points_num = len(data.points)

        # zero centered
        centered_points = data.points - data.sensor_origin
        distances = torch.linalg.norm(centered_points, dim=1, keepdim=True)
        point_dirs = torch.nn.functional.normalize(centered_points, dim=1)

        # around the surface sampling (positive, negative)
        around_surface_sampling_num = self.config.around_surface_sampling_num
        around_surface_sampling_dist = self.config.around_surface_sampling_dist

        around_surface_samples = self.get_samples_between(
            (points_num, around_surface_sampling_num),
            -around_surface_sampling_dist,
            around_surface_sampling_dist,
        )

        # free space sampling (positive)
        free_space_sampling_num = self.config.free_space_sampling_num
        free_space_sampling_ratio = self.config.free_space_sampling_ratio
        free_space_sampling_start_dist = self.config.free_space_sampling_start_dist
        free_space_sampling_dist = distances * free_space_sampling_ratio

        free_space_samples = self.get_samples_between(
            (points_num, free_space_sampling_num),
            torch.full_like(free_space_sampling_dist, free_space_sampling_start_dist),
            free_space_sampling_dist,
        )

        # occupied space sampling (negative)
        occupied_space_sampling_num = self.config.occupied_space_sampling_num
        occupied_space_sampling_dist = self.config.occupied_space_sampling_dist

        occupied_space_samples = self.get_samples_between(
            (points_num, occupied_space_sampling_num),
            -around_surface_sampling_dist - occupied_space_sampling_dist,
            -around_surface_sampling_dist,
        )

        # structure all samples
        all_sample_num = (
            around_surface_sampling_num
            + free_space_sampling_num
            + occupied_space_sampling_num
        )

        all_samples = torch.cat(
            [around_surface_samples, free_space_samples, occupied_space_samples], dim=1
        )

        # get surface mask
        surface_mask = torch.zeros_like(all_samples, dtype=torch.bool)
        surface_mask[:, :around_surface_sampling_num] = True

        # sort samples, closest to sensor origin first
        all_samples, sort_indices = torch.sort(all_samples, dim=1, descending=True)

        surface_mask = torch.take_along_dim(surface_mask, sort_indices, dim=1)

        # prepare the output
        translation = point_dirs.unsqueeze(1).repeat((1, all_sample_num, 1))
        translation = translation * all_samples.unsqueeze(2) * -1

        all_sample_points = centered_points.unsqueeze(1).repeat((1, all_sample_num, 1))
        all_sample_points = all_sample_points + translation

        all_sample_points_translated = all_sample_points + data.sensor_origin

        return LidarSamplingOutput(
            all_sample_points_translated,
            all_samples,
            surface_mask,
        )
