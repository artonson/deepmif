from dataclasses import dataclass

import torch
import torch.nn as nn

from deepmif.model.implicit_network import ImplicitNetwork, ImplicitNetworkConfig
from deepmif.utils import is_none
from deepmif.utils.tools import get_gradient


@dataclass
class MIFDecoderInput:
    sdf: torch.Tensor = None
    points: torch.Tensor = None
    sensor_origins: torch.Tensor = None
    points_init: torch.Tensor = None


@dataclass
class MIFDecoderOutput:
    mif: torch.Tensor = None

    # losses
    ray_monotonicity: torch.Tensor = None
    mif_sign: torch.Tensor = None
    mif_surface: torch.Tensor = None
    gradient: torch.Tensor = None


class MIFDecoder(nn.Module):
    def __init__(self, network_config, feature_vector_size):
        super().__init__()
        self.network_config = network_config

        self.implicit_network = ImplicitNetwork(
            ImplicitNetworkConfig(
                input_dim=feature_vector_size, **network_config.implicit_network
            )
        )

        print(self.implicit_network)

    def forward(self, inputs: MIFDecoderInput, octree):
        if is_none(inputs.sensor_origins):  # predict
            return MIFDecoderOutput(mif=self.implicit(inputs.points, octree))
        else:  # train
            return self.implicit_train(inputs, octree)

    def implicit_train(self, inputs: MIFDecoderInput, octree, alpha=100.0):
        points = inputs.points

        points_flat = points.reshape(-1, 3)
        points_flat.requires_grad = True
        feature_vectors = octree.query_feature(points_flat)

        mif = self.implicit_network(points_flat, feature_vectors)
        mif = mif.reshape(points.shape[:2])

        # monotonic decrease
        ray_monotonicity = (1.0 - torch.tanh(alpha * (mif[:, :1] - mif[:, 1:]))).mean()

        # correct sign
        mif_sign = (1.0 - torch.tanh(alpha * mif) * inputs.sdf.sign()).mean()

        gradient, mif_surface = None, None
        if points_flat.requires_grad and mif.requires_grad:
            init_points = inputs.points_init
            init_points.requires_grad = True
            init_mif = self.implicit(init_points, octree)
            gradient = get_gradient(init_points, init_mif)

            # zero on the surface
            mif_surface = init_mif.abs().mean()

        return MIFDecoderOutput(
            gradient=gradient,
            ray_monotonicity=ray_monotonicity,
            mif_sign=mif_sign,
            mif_surface=mif_surface,
        )

    def implicit(self, points, octree):
        points_flat = points.reshape(-1, 3)
        feature_vectors = octree.query_feature(points_flat)
        return self.implicit_network(points_flat, feature_vectors)
