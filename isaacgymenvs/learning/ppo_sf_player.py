# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import os
import time
import torch
import numpy as np
from rl_games.algos_torch import players
import random
from rl_games.algos_torch import torch_ext
from rl_games.common.tr_helpers import unsqueeze_obs
from rl_games.common.player import BasePlayer
from .sf_player_pool import SFPlayerPool, SFPlayerVectorizedPool, SFPlayerThreadPool, SFPlayerProcessPool, SinglePlayer
import matplotlib.pyplot as plt

from multielo import MultiElo


def rescale_actions(low, high, action):
    d = (high - low) / 2.0
    m = (high + low) / 2.0
    scaled_action = action * d + m
    return scaled_action


class SFPlayer(BasePlayer):
    def __init__(self, params):
        super().__init__(params)
        print(f'params:{params}')
        self.network = self.config['network']
        self.actions_num = self.action_space.shape[0]
        self.actions_low = torch.from_numpy(self.action_space.low.copy()).float().to(self.device)
        self.actions_high = torch.from_numpy(self.action_space.high.copy()).float().to(self.device)
        self.mask = [False]
        self.is_rnn = False
        self.normalize_input = self.config['normalize_input']
        self.normalize_value = self.config.get('normalize_value', False)
        self.base_model_config = {
            'actions_num': self.actions_num,
            'input_shape': self.obs_shape,
            'num_seqs': self.num_agents,
            'value_size': self.env_info.get('value_size', 1),
            'normalize_value': self.normalize_value,
            'normalize_input': self.normalize_input,
        }
        self.policy_timestep = []
        self.policy_op_timestep = []
        self.params = params
        self.record_elo = True
        self.num_actors = params['config']['num_actors']
        self.player_pool_type = params['player_pool_type']
        self.player_pool = None
        self.op_player_pool = None
        self.num_opponents = params['num_agents'] - 1
        self.games_num = 40000
        self.max_steps = 1000
        self.update_op_num = 0
        self.restore_op(params['op_load_path'])
        self.players_per_env = []
        self.elo = MultiElo()
        # assert self.num_actors % self.op_batch_num == 0

    def restore(self, load_dir):
        if os.path.isdir(load_dir):
            self.player_pool = self._build_player_pool(params=self.params, player_num=len(os.listdir(load_dir)))
            print('dir:', load_dir)
            for idx, policy_check_checkpoint in enumerate(os.listdir(load_dir)):
                model = self.load_model(load_dir + '/' + str(policy_check_checkpoint))
                self.policy_timestep.append(os.path.getmtime(load_dir + '/' + str(policy_check_checkpoint)))
                new_player = SinglePlayer(player_idx=policy_check_checkpoint, model=model, device=self.device,
                                          rating=400, obs_batch_len=self.num_actors * self.num_opponents)
                self.player_pool.add_player(new_player)
        else:
            self.player_pool = self._build_player_pool(params=self.params, player_num=1)
            model = self.load_model(load_dir)
            new_player = SinglePlayer(player_idx=0, model=model, device=self.device,
                                      rating=400, obs_batch_len=self.num_actors * self.num_opponents)
            self.player_pool.add_player(new_player)
        self._alloc_env_indices()

    def restore_op(self, load_dir):
        if os.path.isdir(load_dir):
            self.op_player_pool = self._build_player_pool(params=self.params, player_num=len(os.listdir(load_dir)))
            for idx, policy_check_checkpoint in enumerate(os.listdir(load_dir)):
                self.policy_op_timestep.append(os.path.getmtime(load_dir + '/' + str(policy_check_checkpoint)))
                model = self.load_model(load_dir + '/' + str(policy_check_checkpoint))
                new_player = SinglePlayer(player_idx=policy_check_checkpoint, model=model, device=self.device,
                                          rating=400, obs_batch_len=self.num_actors * self.num_opponents)
                self.op_player_pool.add_player(new_player)
        else:
            self.op_player_pool = self._build_player_pool(params=self.params, player_num=1)
            model = self.load_model(load_dir)
            new_player = SinglePlayer(player_idx=0, model=model, device=self.device,
                                      rating=400, obs_batch_len=self.num_actors * self.num_opponents)
            self.op_player_pool.add_player(new_player)

    def _alloc_env_indices(self):
        for idx in range(self.num_actors):
            player_idx = random.randint(0, len(self.player_pool.players) - 1)
            self.player_pool.players[player_idx].add_envs(torch.tensor([idx], dtype=torch.long, device=self.device))
            env_player = [self.player_pool.players[player_idx]]
            for op_idx in range(self.num_opponents):
                op_player_idx = random.randint(0, len(self.op_player_pool.players) - 1)
                self.op_player_pool.players[op_player_idx].add_envs(
                    torch.tensor([idx + op_idx * self.num_actors], dtype=torch.long, device=self.device))
                env_player.append(self.op_player_pool.players[op_player_idx])
            self.players_per_env.append(env_player)
        for player in self.player_pool.players:
            player.reset_envs()
        for player in self.op_player_pool.players:
            player.reset_envs()

    def _build_player_pool(self, params, player_num):

        if self.player_pool_type == 'multi_thread':
            return SFPlayerProcessPool(max_length=player_num,
                                       device=self.device)
        elif self.player_pool_type == 'multi_process':
            return SFPlayerThreadPool(max_length=player_num,
                                      device=self.device)
        elif self.player_pool_type == 'vectorized':
            vector_model_config = self.base_model_config
            vector_model_config['num_envs'] = self.num_actors * self.num_opponents
            vector_model_config['population_size'] = player_num

            return SFPlayerVectorizedPool(max_length=player_num, device=self.device,
                                          vector_model_config=vector_model_config, params=params)
        else:
            return SFPlayerPool(max_length=player_num, device=self.device)

    def _update_rating(self, info, enc_indices):
        for env_idx in enc_indices:
            player = self.players_per_env[env_idx][0]
            op_player = self.players_per_env[env_idx][1]
            if info['win'][env_idx]:
                player.rating, op_player.rating = self.elo.get_new_ratings([player.rating, op_player.rating])
            elif info['lose'][env_idx]:
                op_player.rating, player.rating = self.elo.get_new_ratings([op_player.rating, player.rating])
            elif info['draw'][env_idx]:
                player.rating, op_player.rating = self.elo.get_new_ratings([player.rating, op_player.rating],
                                                                           result_order=[1, 1])

    def run(self):
        n_games = self.games_num
        render = self.render_env
        n_game_life = self.n_game_life
        is_determenistic = self.is_determenistic
        sum_rewards = 0
        sum_steps = 0
        sum_game_res = 0
        n_games = n_games * n_game_life
        games_played = 0
        has_masks = False
        has_masks_func = getattr(self.env, "has_action_mask", None) is not None

        if has_masks_func:
            has_masks = self.env.has_action_mask()
        print(f'games_num:{n_games}')
        need_init_rnn = self.is_rnn
        for _ in range(n_games):
            if games_played >= n_games:
                break

            obses = self.env_reset(self.env)
            batch_size = 1
            batch_size = self.get_batch_size(obses['obs'], batch_size)

            if need_init_rnn:
                self.init_rnn()
                need_init_rnn = False

            cr = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
            steps = torch.zeros(batch_size, dtype=torch.float32, device=self.device)

            print_game_res = False

            for n in range(self.max_steps):
                if has_masks:
                    masks = self.env.get_action_mask()
                    action = self.get_masked_action(
                        obses, masks, is_determenistic)
                else:
                    action = self.get_action(obses['obs'], is_determenistic)
                    action_op = self.get_action(obses['obs_op'], is_determenistic, is_op=True)
                obses, r, done, info = self.env_step(self.env, torch.cat((action, action_op), dim=0))
                cr += r
                steps += 1

                if render:
                    self.env.render(mode='human')
                    time.sleep(self.render_sleep)

                all_done_indices = done.nonzero(as_tuple=False)
                done_indices = all_done_indices[::self.num_agents]
                done_count = len(done_indices)
                games_played += done_count
                if self.record_elo:
                    self._update_rating(info, all_done_indices.flatten())
                if done_count > 0:
                    if self.is_rnn:
                        for s in self.states:
                            s[:, all_done_indices, :] = s[:, all_done_indices, :] * 0.0

                    cur_rewards = cr[done_indices].sum().item()
                    cur_steps = steps[done_indices].sum().item()

                    cr = cr * (1.0 - done.float())
                    steps = steps * (1.0 - done.float())
                    sum_rewards += cur_rewards
                    sum_steps += cur_steps

                    game_res = 0.0
                    if isinstance(info, dict):
                        if 'battle_won' in info:
                            print_game_res = True
                            game_res = info.get('battle_won', 0.5)
                        if 'scores' in info:
                            print_game_res = True
                            game_res = info.get('scores', 0.5)
                    if self.print_stats:
                        if print_game_res:
                            print('reward:', cur_rewards / done_count,
                                  'steps:', cur_steps / done_count, 'w:', game_res)
                        else:
                            print('reward:', cur_rewards / done_count,
                                  'steps:', cur_steps / done_count)

                    sum_game_res += game_res
                    if batch_size // self.num_agents == 1 or games_played >= n_games:
                        break
        # print("winrate:",
        #       self.env.win_count / (self.env.win_count + self.env.lose_count + self.env.draw_count),
        #       "draw:",
        #       self.env.draw_count / (self.env.win_count + self.env.lose_count + self.env.draw_count))

        # print(sum_rewards)
        # if print_game_res:
        #     print('av reward:', sum_rewards / games_played * n_game_life, 'av steps:', sum_steps /
        #           games_played * n_game_life, 'winrate:', sum_game_res / games_played * n_game_life)
        # else:
        #     print('av reward:', sum_rewards / games_played * n_game_life,
        #           'av steps:', sum_steps / games_played * n_game_life)
        if self.record_elo:
            self._plot_elo_curve()

    def _plot_elo_curve(self):
        self.policy_op_timestep.sort()
        self.policy_timestep.sort()
        for idx in range(1, len(self.policy_op_timestep)):
            self.policy_op_timestep[idx] -= self.policy_op_timestep[0]
            self.policy_op_timestep[idx] /= 3600 * 24
            print(self.policy_op_timestep[idx])
        for idx in range(1, len(self.policy_timestep)):
            self.policy_timestep[idx] -= self.policy_timestep[0]
            self.policy_timestep[idx] /= 3600 * 24
        self.policy_timestep[0] = self.policy_op_timestep[0] = 0
        x = np.array(self.policy_timestep)
        y = np.arange(len(self.player_pool.players))
        x_op = np.array(self.policy_op_timestep)
        y_op = np.arange(len(self.op_player_pool.players))
        for player in self.player_pool.players:
            idx = int(player.player_idx.split('_', 1)[1].split('.', 1)[0]) - 1
            # print(player.player_idx, player.rating)
            y[idx] = player.rating
        for player in self.op_player_pool.players:
            idx = int(player.player_idx.split('_', 1)[1].split('.', 1)[0]) - 1
            # print(player.player_idx, player.rating)
            y_op[idx] = player.rating
        l1 = plt.plot(x, y, 'b--', label='policy')
        l2 = plt.plot(x_op, y_op, 'r--', label='policy_op')
        plt.plot(x, y, 'b^-', x_op, y_op, 'ro-')
        plt.title('ELO Curve')
        plt.xlabel('timestep/days')
        plt.ylabel('ElO')
        plt.legend()
        plt.savefig('./elo.jpg')

    def get_action(self, obs, is_determenistic=False, is_op=False):
        if self.has_batch_dimension == False:
            obs = unsqueeze_obs(obs)
        obs = self._preproc_obs(obs)
        input_dict = {
            'is_train': False,
            'prev_actions': None,
            'obs': obs,
            'rnn_states': self.states
        }
        with torch.no_grad():
            data_len = self.num_actors * self.num_opponents if is_op else self.num_actors
            res_dict = {
                "actions": torch.zeros((data_len, self.actions_num), device=self.device),
                "values": torch.zeros((data_len, 1), device=self.device),
                "mus": torch.zeros((data_len, self.actions_num), device=self.device)
            }
            if is_op:
                self.op_player_pool.inference(input_dict, res_dict, obs)
            else:
                self.player_pool.inference(input_dict, res_dict, obs)
        mu = res_dict['mus']
        action = res_dict['actions']
        # self.states = res_dict['rnn_states']
        if is_determenistic:
            current_action = mu
        else:
            current_action = action
        if self.has_batch_dimension == False:
            current_action = torch.squeeze(current_action.detach())

        if self.clip_actions:
            return rescale_actions(self.actions_low, self.actions_high, torch.clamp(current_action, -1.0, 1.0))
        else:
            return current_action

    def env_reset(self, env):
        obs = env.reset()
        obs['obs_op'] = obs['obs'][self.num_actors:]
        obs['obs'] = obs['obs'][:self.num_actors]
        return obs

    def env_step(self, env, actions):
        obs, rewards, dones, infos = env.step(actions)
        obs['obs_op'] = obs['obs'][self.num_actors:]
        obs['obs'] = obs['obs'][:self.num_actors]
        if hasattr(obs, 'dtype') and obs.dtype == np.float64:
            obs = np.float32(obs)
        if self.value_size > 1:
            rewards = rewards[0]
        if self.is_tensor_obses:
            return self.obs_to_torch(obs), rewards.cpu(), dones.cpu(), infos
        else:
            if np.isscalar(dones):
                rewards = np.expand_dims(np.asarray(rewards), 0)
                dones = np.expand_dims(np.asarray(dones), 0)
            return obs, rewards, dones, infos

    def create_model(self):
        model = self.network.build(self.base_model_config)
        model.to(self.device)
        return model

    def load_model(self, fn):
        model = self.create_model()
        checkpoint = torch_ext.safe_filesystem_op(torch.load, fn, map_location=self.device)
        model.load_state_dict(checkpoint['model'])
        if self.normalize_input and 'running_mean_std' in checkpoint:
            model.running_mean_std.load_state_dict(checkpoint['running_mean_std'])
        return model
