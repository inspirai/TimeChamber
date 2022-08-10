import torch
import torch.nn as nn
import time
import math
import numpy as np
import torch.nn.functional as F
import random
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from trueskill import Rating, rate_1vs1
from multielo import MultiElo
from torch.profiler import profile, record_function, ProfilerActivity


# a = torch.nn.Linear(40, 8, device='cuda:0')


# b = torch.tensor([[[1, 2, 3, 4], [1, 2, 3, 4]], [[4, 2, 3, 4], [2, 3, 4, 6]]], dtype=torch.float32,)
class VectorizedLinearLayer(torch.nn.Module):
    """Vectorized version of torch.nn.Linear."""

    def __init__(
            self,
            population_size: int,
            in_features: int,
            out_features: int,
            use_layer_norm: bool = False,
    ):
        super().__init__()
        self._population_size = population_size
        self._in_features = in_features
        self._out_features = out_features
        self.bias = torch.nn.Parameter(
            torch.empty(self._population_size, 1, self._out_features),
            requires_grad=True,
        )

        self.weight = torch.nn.Parameter(
            torch.empty(self._population_size, self._in_features, self._out_features),
            requires_grad=True,
        )

        for member_id in range(population_size):
            torch.nn.init.kaiming_uniform_(self.weight[member_id], a=math.sqrt(5))
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight[0])
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        torch.nn.init.uniform_(self.bias, -bound, bound)

        self._layer_norm = (
            torch.nn.LayerNorm(self._out_features, self._population_size)
            if use_layer_norm
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[0] == self._population_size
        if self._layer_norm is not None:
            return self._layer_norm(x.matmul(self.weight) + self.bias)
        return x.matmul(self.weight) + self.bias


def func():
    in_size = 40
    units = [8, 128, 8]
    layers = []
    for unit in units:
        layers.append(
            VectorizedLinearLayer(32, in_size, unit, False))
        layers.append(nn.ReLU())
        in_size = unit
    return nn.Sequential(*layers)


def func2():
    in_size = 40
    units = [256, 128, 64]
    layers = []
    for unit in units:
        layers.append(
            nn.Linear(in_size, unit, device='cuda:0'))
        layers.append(nn.ELU())
        in_size = unit
    return nn.Sequential(*layers)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)  # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
net.to('cuda:0')

# a = VectorizedLinearLayer(in_features=40, out_features=8, population_size=16, use_layer_norm=False)
a = func()
model_mlps = []
for _ in range(32):
    model = func2()
    model.to('cuda:0')
    model_mlps.append(model)
a.to('cuda:0')
data = torch.rand((4096, 40), dtype=torch.float32, device='cuda:0')
b = torch.rand((32, 4096, 40), dtype=torch.float32, device='cuda:0')
sigma = nn.Parameter(torch.zeros((32, 1, 8,), requires_grad=True, dtype=torch.float32, device='cuda:0'),
                     requires_grad=True)

a = torch.tensor([[1, 2, 3], [3, 2, 1]])
b = torch.sum(a[:, 1] == 2)
print(b)
