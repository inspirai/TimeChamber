import torch
import torch.nn as nn
from rl_games.algos_torch.running_mean_std import RunningMeanStd, RunningMeanStdObs
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.models import ModelA2CContinuousLogStd


class VectorizedRunningMeanStd(RunningMeanStd):
    def __init__(self, insize, population_size, epsilon=1e-05, per_channel=False, norm_only=False, is_training=False):
        # input shape: population_size*batch_size*(insize)
        super(VectorizedRunningMeanStd, self).__init__(population_size, epsilon, per_channel, norm_only)
        self.insize = insize
        self.epsilon = epsilon
        self.population_size = population_size
        self.training = is_training
        self.norm_only = norm_only
        self.per_channel = per_channel
        if per_channel:
            if len(self.insize) == 3:
                self.axis = [1, 3, 4]
            if len(self.insize) == 2:
                self.axis = [1, 3]
            if len(self.insize) == 1:
                self.axis = [1]
            in_size = self.insize[1]
        else:
            self.axis = [1]
            in_size = insize
        # print(in_size)
        self.register_buffer("running_mean", torch.zeros((population_size, *in_size), dtype=torch.float32))
        self.register_buffer("running_var", torch.ones((population_size, *in_size), dtype=torch.float32))
        self.register_buffer("count", torch.ones((population_size, 1), dtype=torch.float32))

    def _update_mean_var_count_from_moments(self, mean, var, count, batch_mean, batch_var, batch_count):
        delta = batch_mean - mean
        tot_count = count + batch_count
        new_mean = mean + delta * batch_count / tot_count
        m_a = var * count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta ** 2 * count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count
        return new_mean, new_var, new_count

    def forward(self, input, unnorm=False, mask=None):
        if self.training:
            if mask is not None:
                mean, var = torch_ext.get_mean_std_with_masks(input, mask)
            else:
                mean = input.mean(self.axis)  # along channel axis
                var = input.var(self.axis)
            self.running_mean, self.running_var, self.count = self._update_mean_var_count_from_moments(
                self.running_mean, self.running_var, self.count,
                mean, var, input.size()[1])

        # change shape
        if self.per_channel:
            if len(self.insize) == 3:
                current_mean = self.running_mean.view([self.population_size, 1, self.insize[0], 1, 1]).expand_as(input)
                current_var = self.running_var.view([self.population_size, 1, self.insize[0], 1, 1]).expand_as(input)
            if len(self.insize) == 2:
                current_mean = self.running_mean.view([self.population_size, 1, self.insize[0], 1]).expand_as(input)
                current_var = self.running_var.view([self.population_size, 1, self.insize[0], 1]).expand_as(input)
            if len(self.insize) == 1:
                current_mean = self.running_mean.view([self.population_size, 1, self.insize[0]]).expand_as(input)
                current_var = self.running_var.view([self.population_size, 1, self.insize[0]]).expand_as(input)
        else:
            current_mean = self.running_mean
            current_var = self.running_var
        # get output

        if unnorm:
            y = torch.clamp(input, min=-5.0, max=5.0)
            y = torch.sqrt(torch.unsqueeze(current_var.float(), 1) + self.epsilon) * y + torch.unsqueeze(
                current_mean.float(), 1)
        else:
            if self.norm_only:
                y = input / torch.sqrt(current_var.float() + self.epsilon)
            else:
                y = (input - torch.unsqueeze(current_mean.float(), 1)) / torch.sqrt(
                    torch.unsqueeze(current_var.float(), 1) + self.epsilon)
                y = torch.clamp(y, min=-5.0, max=5.0)
        return y


class ModelVectorizedA2C(ModelA2CContinuousLogStd):
    def __init__(self, network):
        super().__init__(network)
        return

    def build(self, config):
        net = self.network_builder.build('vectorized_a2c', **config)
        for name, _ in net.named_parameters():
            print(name)

        obs_shape = config['input_shape']
        population_size = config['population_size']
        normalize_value = config.get('normalize_value', False)
        normalize_input = config.get('normalize_input', False)
        value_size = config.get('value_size', 1)

        return self.Network(net, population_size, obs_shape=obs_shape,
                            normalize_value=normalize_value, normalize_input=normalize_input, value_size=value_size, )

    class Network(ModelA2CContinuousLogStd.Network):
        def __init__(self, a2c_network, population_size, obs_shape, normalize_value, normalize_input, value_size):
            self.population_size = population_size
            super().__init__(a2c_network, obs_shape=obs_shape,
                             normalize_value=normalize_value, normalize_input=normalize_input, value_size=value_size)
            if normalize_value:
                self.value_mean_std = VectorizedRunningMeanStd((self.value_size,), self.population_size)
            if normalize_input:
                if isinstance(obs_shape, dict):
                    self.running_mean_std = RunningMeanStdObs(obs_shape)
                else:
                    self.running_mean_std = VectorizedRunningMeanStd(obs_shape, self.population_size)

        def update(self, population_idx, network):
            for key in self.state_dict():
                param1 = self.state_dict()[key]
                param2 = network.state_dict()[key]
                if len(param1.shape) == len(param2.shape):
                    self.state_dict()[key] = param2
                elif len(param2.shape) == 1:
                    if len(param1.shape) == 3:
                        param1[population_idx] = torch.unsqueeze(param2, dim=0)
                    else:
                        param1[population_idx] = param2
                elif len(param2.shape) == 2:
                    param1[population_idx] = torch.transpose(param2, 0, 1)
