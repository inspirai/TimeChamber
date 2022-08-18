import collections

import random
import torch
import torch.multiprocessing as mp
import dill
# import time
from rl_games.algos_torch import model_builder
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, ALL_COMPLETED


def player_inference_thread(model, input_dict, res_dict, env_indices, processed_obs):
    if len(env_indices) == 0:
        return None
    input_dict['obs'] = processed_obs[env_indices]
    out_dict = model(input_dict)
    for key in res_dict:
        res_dict[key][env_indices] = out_dict[key]
    return out_dict


def player_inference_process(pipe, queue, barrier):
    input_dict = {
        'is_train': False,
        'prev_actions': None,
        'obs': None,
        'rnn_states': None,
    }
    model = None
    barrier.wait()
    while True:
        msg = pipe.recv()
        task = msg['task']
        if task == 'init':
            if model is not None:
                del model
            model = queue.get()
            model = dill.loads(model)
            barrier.wait()
        elif task == 'forward':
            obs, actions, values, env_indices = queue.get()
            input_dict['obs'] = obs[env_indices]
            out_dict = model(input_dict)
            actions[env_indices] = out_dict['actions']
            values[env_indices] = out_dict['values']
            barrier.wait()
            del obs, actions, values, env_indices
        elif task == 'terminate':
            break
        else:
            barrier.wait()


class SinglePlayer:
    def __init__(self, player_idx, model, device, obs_batch_len=0, rating=None):
        self.model = model
        if model:
            self.model.eval()
        self.player_idx = player_idx
        self._games = torch.tensor(0, device=device, dtype=torch.float)
        self._wins = torch.tensor(0, device=device, dtype=torch.float)
        self._loses = torch.tensor(0, device=device, dtype=torch.float)
        self._draws = torch.tensor(0, device=device, dtype=torch.float)
        self._decay = 0.998
        self._has_env = torch.zeros((obs_batch_len,), device=device, dtype=torch.bool)
        self.device = device
        self.env_indices = torch.tensor([], device=device, dtype=torch.long, requires_grad=False)
        if rating:
            self.rating = rating

    def __call__(self, input_dict):
        return self.model(input_dict)

    def reset_envs(self):
        self.env_indices = self._has_env.nonzero(as_tuple=True)

    def remove_envs(self, env_indices):
        self._has_env[env_indices] = False

    def add_envs(self, env_indices):
        self._has_env[env_indices] = True

    def clear_envs(self):
        self.env_indices = torch.tensor([], device=self.device, dtype=torch.long, requires_grad=False)

    def update_metric(self, wins, loses, draws):
        win_count = torch.sum(wins[self.env_indices])
        lose_count = torch.sum(loses[self.env_indices])
        draw_count = torch.sum(draws[self.env_indices])
        for stats in (self._games, self._wins, self._loses, self._draws):
            stats *= self._decay
        self._games += win_count + lose_count + draw_count
        self._wins += win_count
        self._loses += lose_count
        self._draws += draw_count

    def clear_metric(self):
        self._games = torch.tensor(0, device=self.device, dtype=torch.float)
        self._wins = torch.tensor(0, device=self.device, dtype=torch.float)
        self._loses = torch.tensor(0, device=self.device, dtype=torch.float)
        self._draws = torch.tensor(0, device=self.device, dtype=torch.float)

    def win_rate(self):
        if self.model is None:
            return 0
        elif self._games == 0:
            return 0.5
        return (self._wins + 0.5 * self._draws) / self._games

    def games_num(self):
        return self._games


class SFPlayerPool:
    def __init__(self, max_length, device):
        assert max_length > 0
        self.players = []
        self.max_length = max_length
        self.idx = 0
        self.device = device
        self.weightings = {
            "variance": lambda x: x * (1 - x),
            "linear": lambda x: 1 - x,
            "squared": lambda x: (1 - x) ** 2,
        }

    def add_player(self, player):
        if len(self.players) < self.max_length:
            self.players.append(player)
        else:
            self.players[self.idx] = player
        self.idx += 1
        self.idx %= self.max_length

    def sample_player(self, weight='linear'):
        weight_func = self.weightings[weight]
        player = \
            random.choices(self.players, weights=[weight_func(player.win_rate()) for player in self.players])[0]
        return player

    def update_player_metric(self, infos):
        for player in self.players:
            player.update_metric(infos['win'], infos['lose'], infos['draw'])

    def clear_player_metric(self):
        for player in self.players:
            player.clear_metric()

    def inference(self, input_dict, res_dict, processed_obs):
        for i, player in enumerate(self.players):
            if len(player.env_indices) == 0:
                continue
            input_dict['obs'] = processed_obs[player.env_indices]
            out_dict = player(input_dict)
            for key in res_dict:
                res_dict[key][player.env_indices] = out_dict[key]


class SFPlayerVectorizedPool(SFPlayerPool):
    def __init__(self, max_length, device, vector_model_config, params):
        super(SFPlayerVectorizedPool, self).__init__(max_length, device)
        params['model']['name'] = 'vectorized_a2c'
        params['network']['name'] = 'vectorized_a2c'
        builder = model_builder.ModelBuilder()
        self.vectorized_network = builder.load(params)
        self.vectorized_model = self.vectorized_network.build(vector_model_config)
        self.vectorized_model.to(self.device)
        self.vectorized_model.eval()
        self.obs = torch.zeros(
            (self.max_length, vector_model_config["num_envs"], vector_model_config['input_shape'][0]),
            dtype=torch.float32, device=self.device)
        for idx in range(max_length):
            self.add_player(SinglePlayer(idx, None, self.device, vector_model_config["num_envs"]))

    def inference(self, input_dict, res_dict, processed_obs):
        for i, player in enumerate(self.players):
            self.obs[i][player.env_indices] = processed_obs[player.env_indices]
        input_dict['obs'] = self.obs
        out_dict = self.vectorized_model(input_dict)
        for i, player in enumerate(self.players):
            if len(player.env_indices) == 0:
                continue
            for key in res_dict:
                res_dict[key][player.env_indices] = out_dict[key][i][player.env_indices]

    def add_player(self, player):
        if player.model:
            self.vectorized_model.update(self.idx, player.model)
        super().add_player(player)


class SFPlayerThreadPool(SFPlayerPool):
    def __init__(self, max_length, device):
        super().__init__(max_length, device)
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_length)

    def inference(self, input_dict, res_dict, processed_obs):
        self.thread_pool.map(player_inference_thread, [player.model for player in self.players],
                             [input_dict for _ in range(len(self.players))],
                             [res_dict for _ in range(len(self.players))],
                             [player.env_indices for player in self.players],
                             [processed_obs for _ in range(len(self.players))])


class SFPlayerProcessPool(SFPlayerPool):
    def __init__(self, max_length, device):
        super(SFPlayerProcessPool, self).__init__(max_length, device)
        self.inference_processes = []
        self.queues = []
        self.producer_pipes = []
        self.consumer_pipes = []
        self.barrier = mp.Barrier(self.max_length + 1)
        mp.set_start_method(method='spawn', force=True)
        self._init_inference_processes()

    def _init_inference_processes(self):
        for _ in range(self.max_length):
            queue = mp.Queue()
            self.queues.append(queue)
            pipe_read, pipe_write = mp.Pipe(duplex=False)
            self.producer_pipes.append(pipe_write)
            self.consumer_pipes.append(pipe_read)
            process = mp.Process(target=player_inference_process,
                                 args=(pipe_read, queue, self.barrier),
                                 daemon=True)
            self.inference_processes.append(process)
            process.start()
        self.barrier.wait()

    def add_player(self, player):
        with torch.no_grad():
            model = dill.dumps(player.model)
            for i in range(self.max_length):
                if i == self.idx:
                    self.producer_pipes[i].send({'task': 'init'})
                    self.queues[i].put(model)
                else:
                    self.producer_pipes[i].send({'task': 'continue'})
            self.barrier.wait()
            if len(self.players) < self.max_length:
                self.players.append(player)
            else:
                self.players[self.idx] = player
            self.idx += 1
            self.idx %= self.max_length

    def inference(self, input_dict, res_dict, processed_obs):

        for i in range(self.max_length):
            if i < len(self.players) and len(self.players[i].env_indices):
                self.producer_pipes[i].send({'task': 'forward'})
                self.queues[i].put(
                    (processed_obs, res_dict['actions'],
                     res_dict['values'], self.players[i].env_indices))
            else:
                self.producer_pipes[i].send({'task': 'continue'})

    def __del__(self):
        for pipe in self.producer_pipes:
            pipe.send({'task': 'terminate'})
        for process in self.inference_processes:
            process.join()
