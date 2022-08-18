import torch
import torch.nn as nn
import math
from rl_games.algos_torch import network_builder


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

        self.weight = torch.nn.Parameter(
            torch.empty(self._population_size, self._in_features, self._out_features),
            requires_grad=True,
        )
        self.bias = torch.nn.Parameter(
            torch.empty(self._population_size, 1, self._out_features),
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


class VectorizedA2CBuilder(network_builder.A2CBuilder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    class Network(network_builder.A2CBuilder.Network):
        def __init__(self, params, **kwargs):
            self.population_size = kwargs.get('population_size')
            super().__init__(params, **kwargs)

            self.value = VectorizedLinearLayer(population_size=self.population_size,
                                               in_features=self.units[-1],
                                               out_features=self.value_size)
            actions_num = kwargs.get('actions_num')
            self.mu = VectorizedLinearLayer(self.population_size, self.units[-1], actions_num)
            if self.fixed_sigma:
                self.sigma = nn.Parameter(
                    torch.zeros((self.population_size, 1, actions_num), requires_grad=True, dtype=torch.float32),
                    requires_grad=True)
            else:
                self.sigma = VectorizedLinearLayer(self.population_size, self.units[-1], actions_num)

        def _build_vectorized_mlp(self,
                                  input_size,
                                  units,
                                  activation,
                                  norm_func_name=None):
            print(f'build vectorized mlp:{self.population_size}x{input_size}')
            in_size = input_size
            layers = []
            for unit in units:
                layers.append(
                    VectorizedLinearLayer(self.population_size, in_size, unit, norm_func_name == 'layer_norm'))
                layers.append(self.activations_factory.create(activation))
                in_size = unit
            return nn.Sequential(*layers)

        def _build_mlp(self,
                       input_size,
                       units,
                       activation,
                       dense_func,
                       norm_only_first_layer=False,
                       norm_func_name=None,
                       d2rl=False):
            return self._build_vectorized_mlp(input_size, units, activation, norm_func_name=norm_func_name)

        def forward(self, obs_dict):  # implement continues situation
            obs = obs_dict['obs']
            states = obs_dict.get('rnn_states', None)
            out = self.actor_mlp(obs)
            value = self.value_act(self.value(out))
            mu = self.mu_act(self.mu(out))
            if self.fixed_sigma:
                sigma = self.sigma_act(self.sigma)
            else:
                sigma = self.sigma_act(self.sigma(out))
            return mu, mu * 0 + sigma, value, states

        def load(self, params):
            super().load(params)

    def build(self, name, **kwargs):
        net = VectorizedA2CBuilder.Network(self.params, **kwargs)
        return net
