from dataclasses import dataclass

import torch
import torch.nn as nn

from deepmif.model.embedder import get_embedder
from deepmif.model.mlp import MLP


@dataclass(kw_only=True)
class ImplicitNetworkConfig:
    input_dim: int = 13

    multires_point: int = 10

    layers: list = None  # [256, 256, 256, 256]


class ImplicitNetwork(nn.Module):
    def __init__(self, conf: ImplicitNetworkConfig):
        super().__init__()

        # common
        self.input_dim = conf.input_dim + 3

        self.embedpoint_fn = lambda x: x

        if conf.multires_point > 0:
            self.embedpoint_fn, input_point_ch = get_embedder(conf.multires_point)
            self.input_dim += input_point_ch - 3

        self.layers = MLP([self.input_dim] + conf.layers + [1])

    def forward(self, points, feature_vectors):
        p_e = self.embedpoint_fn(points)
        net_inp = torch.cat([p_e, feature_vectors], dim=-1)

        return self.layers(net_inp).flatten()
