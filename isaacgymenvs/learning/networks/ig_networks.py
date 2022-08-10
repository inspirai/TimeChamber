from rl_games.algos_torch import network_builder

import torch
import torch.nn as nn
import numpy as np

from .transformers.utils.transformers import TransformerClassifier

class EncoderMLPBuilder(network_builder.NetworkBuilder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    def load(self, params):
        self.params = params

    def build(self, name, **kwargs):
        net = EncoderMLPBuilder.Network(self.params, **kwargs)
        return net

    class Network(network_builder.NetworkBuilder.BaseNetwork):
        def __init__(self, params, **kwargs):
            actions_num = kwargs.pop('actions_num')
            input_shape = kwargs.pop('input_shape')
            self.value_size = kwargs.pop('value_size', 1)
            network_builder.NetworkBuilder.BaseNetwork.__init__(self)
            self.load(params)

            if self.embedding_reduction == 'sum' :
                self.red_op = torch.sum
            if self.embedding_reduction == 'mean' :
                self.red_op = torch.mean
            self.mlp = nn.Sequential()
            self.encoders = torch.nn.ModuleList([torch.nn.Linear(num, self.embedding_size,bias=False) for num in self.input_split])
            mlp_input_shape = self.embedding_size

            out_size = self.units[-1]

            self.encoder_act = self.activations_factory.create(self.activation) 
            mlp_args = {
                'input_size' : mlp_input_shape, 
                'units' : self.units, 
                'activation' : self.activation, 
                'norm_func_name' : self.normalization,
                'dense_func' : torch.nn.Linear,
                'd2rl' : self.is_d2rl,
                'norm_only_first_layer' : self.norm_only_first_layer
            }
            self.mlp = self._build_mlp(**mlp_args)

            self.value = torch.nn.Linear(out_size, self.value_size)
            self.value_act = self.activations_factory.create(self.value_activation)
            if self.is_continuous:
                self.mu = torch.nn.Linear(out_size, actions_num)
                self.mu_act = self.activations_factory.create(self.space_config['mu_activation']) 
                mu_init = self.init_factory.create(**self.space_config['mu_init'])
                self.sigma_act = self.activations_factory.create(self.space_config['sigma_activation']) 
                sigma_init = self.init_factory.create(**self.space_config['sigma_init'])

                if self.fixed_sigma:
                    self.sigma = nn.Parameter(torch.zeros(actions_num, requires_grad=True, dtype=torch.float32), requires_grad=True)
                else:
                    self.sigma = torch.nn.Linear(out_size, actions_num)

            mlp_init = self.init_factory.create(**self.initializer)

            for m in self.modules():         
                if isinstance(m, nn.Linear):
                    mlp_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)    

            if self.is_continuous:
                mu_init(self.mu.weight)
                if self.fixed_sigma:
                    sigma_init(self.sigma)
                else:
                    sigma_init(self.sigma.weight)  


        def load(self, params):
            super().load(params)

            return

        def is_separate_critic(self):
            return False

        def is_rnn(self):
            return False

        def get_default_rnn_state(self):
            return None
             
        def load(self, params):
            self.embedding_size = params['mlp']['embedding_size']
            self.embedding_reduction = params['mlp']['embedding_reduction']
            self.input_split = list(params['mlp']['input_split'])

            self.units = params['mlp']['units']
            self.activation = params['mlp']['activation']
            self.initializer = params['mlp']['initializer']
            self.is_d2rl = params['mlp'].get('d2rl', False)
            self.norm_only_first_layer = params['mlp'].get('norm_only_first_layer', False)
            self.value_activation = params.get('value_activation', 'None')
            self.normalization = params.get('normalization', None)
            self.has_space = 'space' in params
            self.central_value = params.get('central_value', False)
            self.is_continuous = True
            self.space_config = params['space']['continuous']
            self.fixed_sigma = self.space_config['fixed_sigma']            

        def forward(self, obs_dict):
            obs = obs_dict['obs']
            outs = torch.split(obs, self.input_split, dim=1)
            encoded_features = [e(o) for e, o in zip(self.encoders, outs)]
            
            stacked_features = torch.stack(encoded_features)
            out = self.red_op(stacked_features, dim=0)
            out = self.encoder_act(out)
            out = self.mlp(out)
            value = self.value_act(self.value(out))

            if self.central_value:
                return value, None


            if self.is_continuous:
                mu = self.mu_act(self.mu(out))
                if self.fixed_sigma:
                    sigma = self.sigma_act(self.sigma)
                else:
                    sigma = self.sigma_act(self.sigma(out))
                return mu, mu*0 + sigma, value, None


class TransformerModel(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self,
    actions_num = 1,
    input_split = [], 
    seq_pool=True,
    embedding_dim=768,
    num_layers=12,
    num_heads=12,
    mlp_ratio=4.0,
    dropout=0.1,
    attention_dropout=0.1,
    stochastic_depth=0.1,
    positional_embedding='none',
    ):
        super(TransformerModel, self).__init__()
        
        self.input_split = list(input_split)
        self.encoders = torch.nn.ModuleList([torch.nn.Linear(num, embedding_dim,bias=False) for num in self.input_split])
        
        self.actions_num = actions_num
        self.sigma = nn.Parameter(torch.zeros(actions_num, requires_grad=True, dtype=torch.float32), requires_grad=True)
        self.transformer_encoder = TransformerClassifier(
            seq_pool=seq_pool,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            num_classes=actions_num+1,
            dropout=dropout,
            attention_dropout=attention_dropout,
            stochastic_depth=stochastic_depth,
            positional_embedding=positional_embedding,
            sequence_length= len(self.input_split)
        )
        #for m in self.encoders:
        #    nn.init.trunc_normal_(m.weight, std=.02)


    def forward(self, src):
        src = torch.split(src, self.input_split, dim=1)
        src = [e(o) for e, o in zip(self.encoders, src)]
        src = [e.unsqueeze(1) for e in src]
        src = torch.cat(src, dim=1)
        output = self.transformer_encoder(src)
        mu, value = torch.split(output, [self.actions_num,1], dim=1)
        return mu, mu*0 + self.sigma, value, None

class TransformerBuilder(network_builder.NetworkBuilder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    def load(self, params):
        self.params = params

    def build(self, name, **kwargs):
        net = TransformerBuilder.Network(self.params, **kwargs)
        return net

    class Network(network_builder.NetworkBuilder.BaseNetwork):
        def __init__(self, params, **kwargs):
            actions_num = kwargs.pop('actions_num')
            input_shape = kwargs.pop('input_shape')
            self.value_size = kwargs.pop('value_size', 1)
            network_builder.NetworkBuilder.BaseNetwork.__init__(self)
            self.load(params)

            self.transformer = TransformerModel(actions_num, **self.tranformer_params)
            

        def load(self, params):
            super().load(params)

            return

        def is_separate_critic(self):
            return False

        def is_rnn(self):
            return False

        def get_default_rnn_state(self):
            return None
             
        def load(self, params):
            self.tranformer_params = params['transformer']        

        def forward(self, obs_dict):
            obs = obs_dict['obs']
            return self.transformer(obs)