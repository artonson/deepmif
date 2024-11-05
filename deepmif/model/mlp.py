from typing import List

from torch import nn


class MLP(nn.Module):
    def __init__(
        self,
        hidden_sizes: List[int],
        act_fn: nn.Module = nn.LeakyReLU,
    ):
        super().__init__()

        self.hidden_sizes = hidden_sizes
        self.act_fn = act_fn

        self.layers = nn.Sequential(*self.__build_layers())

    def linear(self, in_dim, out_dim):
        lin = nn.Linear(in_dim, out_dim, True)
        return nn.utils.weight_norm(lin)

    def __build_layers(self):
        if len(self.hidden_sizes) < 2:
            raise RuntimeError("Bad MLP params!")

        if len(self.hidden_sizes) == 2:
            return [self.linear(self.hidden_sizes[0], self.hidden_sizes[1])]

        layers = []
        for in_size, out_size in zip(self.hidden_sizes[:-1], self.hidden_sizes[1:]):
            layers.append(self.linear(in_size, out_size))
            layers.append(self.act_fn())

        layers.pop()
        return layers

    def forward(self, x):
        return self.layers(x)
