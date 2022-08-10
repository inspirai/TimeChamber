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

import copy
from datetime import datetime
from gym import spaces
import numpy as np
import os
import time
from .sf_player_pool import SFPlayerPool, SinglePlayer, SFPlayerThreadPool, SFPlayerProcessPool, SFPlayerVectorizedPool
from rl_games.algos_torch import a2c_continuous
from rl_games.common.a2c_common import swap_and_flatten01
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch import central_value
import torch
from torch import optim
from tensorboardX import SummaryWriter
import torch.distributed as dist


class SFAgent(a2c_continuous.A2CAgent):
    def __init__(self, base_name, params):
        params['config']['device'] = params['device']
        super().__init__(base_name, params)
        self.player_pool_type = params['player_pool_type']
        self.base_model_config = {
            'actions_num': self.actions_num,
            'input_shape': self.obs_shape,
            'num_seqs': self.num_agents,
            'value_size': self.env_info.get('value_size', 1),
            'normalize_value': self.normalize_value,
            'normalize_input': self.normalize_input,
        }
        self.max_his_player_num = 4
        self.player_pool = self._build_player_pool(params)
        if params['op_load_path']:
            self.op_model = self.create_model()
            self.restore_op(params['op_load_path'])
        else:
            self.op_model = self.model
        self.save_freq = 1000
        self.update_win_rate = 0.9
        self.games_to_check = 10000
        self.update_op_num = 0
        self.op_batch_num = 32
        self.fixed_env_num = 0
        self.update_player_pool(self.model, player_idx=self.update_op_num)
        assert self.num_actors % self.op_batch_num == 0
        assert self.num_actors % self.max_his_player_num == 0

    def _build_player_pool(self, params):
        if self.player_pool_type == 'multi_thread':
            return SFPlayerProcessPool(max_length=self.max_his_player_num,
                                       device=self.device)
        elif self.player_pool_type == 'multi_process':
            return SFPlayerThreadPool(max_length=self.max_his_player_num,
                                      device=self.device)
        elif self.player_pool_type == 'vectorized':
            vector_model_config = self.base_model_config
            vector_model_config['num_envs'] = self.num_actors
            vector_model_config['population_size'] = self.max_his_player_num

            return SFPlayerVectorizedPool(max_length=self.max_his_player_num, device=self.device,
                                          vector_model_config=vector_model_config, params=params)
        else:
            return SFPlayerPool(max_length=self.max_his_player_num, device=self.device)

    def play_steps(self):
        update_list = self.update_list

        step_time = 0.0
        for n in range(self.horizon_length):
            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.obs, masks)
            else:
                step_time_start = time.time()
                res_dict_op = self.get_action_values(self.obs, is_op=True)
                step_time_end = time.time()
                res_dict = self.get_action_values(self.obs)
            self.experience_buffer.update_data('obses', n, self.obs['obs'])
            self.experience_buffer.update_data('dones', n, self.dones)
            for k in update_list:
                self.experience_buffer.update_data(k, n, res_dict[k])
            if self.has_central_value:
                self.experience_buffer.update_data('states', n, self.obs['states'])

            # torch.cuda.synchronize(self.device)
            # print(res_dict_op['actions'])
            if self.player_pool_type == 'multi_thread':
                self.player_pool.thread_pool.shutdown()

            self.obs, rewards, self.dones, infos = self.env_step(
                torch.cat((res_dict['actions'], res_dict_op['actions']), dim=0))

            step_time += (step_time_end - step_time_start)

            shaped_rewards = self.rewards_shaper(rewards)
            if self.value_bootstrap and 'time_outs' in infos:
                shaped_rewards += self.gamma * res_dict['values'] * self.cast_obs(infos['time_outs']).unsqueeze(
                    1).float()

            self.experience_buffer.update_data('rewards', n, shaped_rewards)

            self.current_rewards += rewards
            self.current_lengths += 1
            all_done_indices = self.dones.nonzero(as_tuple=False)
            env_done_indices = self.dones.view(self.num_actors, self.num_agents).all(dim=1).nonzero(as_tuple=False)

            self.game_rewards.update(self.current_rewards[env_done_indices])
            self.game_lengths.update(self.current_lengths[env_done_indices])
            self.algo_observer.process_infos(infos, env_done_indices)

            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones

            for player in self.player_pool.players:
                player.update(wins=infos['lose'], loses=infos['win'], draws=infos['draw'])
        # print(step_time)
        last_values = self.get_values(self.obs)

        fdones = self.dones.float()
        mb_fdones = self.experience_buffer.tensor_dict['dones'].float()
        mb_values = self.experience_buffer.tensor_dict['values']
        mb_rewards = self.experience_buffer.tensor_dict['rewards']
        mb_advs = self.discount_values(fdones, last_values, mb_fdones, mb_values, mb_rewards)
        mb_returns = mb_advs + mb_values

        batch_dict = self.experience_buffer.get_transformed_list(swap_and_flatten01, self.tensor_list)
        batch_dict['returns'] = swap_and_flatten01(mb_returns)
        batch_dict['played_frames'] = self.batch_size
        batch_dict['step_time'] = step_time
        return batch_dict

    def train(self):
        self.init_tensors()
        self.mean_rewards = self.last_mean_rewards = -100500
        start_time = time.time()
        total_time = 0
        rep_count = 0
        # self.frame = 0  # loading from checkpoint
        self.obs = self.env_reset()

        if self.multi_gpu:
            torch.cuda.set_device(self.rank)
            print("====================broadcasting parameters")
            model_params = [self.model.state_dict()]
            dist.broadcast_object_list(model_params, 0)
            self.model.load_state_dict(model_params[0])

        while True:
            epoch_num = self.update_epoch()
            step_time, play_time, update_time, sum_time, a_losses, c_losses, b_losses, entropies, kls, last_lr, lr_mul = self.train_epoch()
            # cleaning memory to optimize space
            self.dataset.update_values_dict(None)
            total_time += sum_time
            curr_frames = self.curr_frames * self.rank_size if self.multi_gpu else self.curr_frames
            self.frame += curr_frames
            should_exit = False

            if self.rank == 0:
                self.diagnostics.epoch(self, current_epoch=epoch_num)
                scaled_time = self.num_agents * sum_time
                scaled_play_time = self.num_agents * play_time

                frame = self.frame // self.num_agents

                if self.print_stats:
                    step_time = max(step_time, 1e-6)
                    fps_step = curr_frames / step_time
                    fps_step_inference = curr_frames / scaled_play_time
                    fps_total = curr_frames / scaled_time
                    print(
                        f'fps step: {fps_step:.0f} fps step and policy inference: {fps_step_inference:.0f} fps total: {fps_total:.0f} epoch: {epoch_num}/{self.max_epochs}')

                self.write_stats(total_time, epoch_num, step_time, play_time, update_time, a_losses, c_losses,
                                 entropies, kls, last_lr, lr_mul, frame, scaled_time, scaled_play_time, curr_frames)

                self.algo_observer.after_print_stats(frame, epoch_num, total_time)

                if self.game_rewards.current_size > 0:
                    mean_rewards = self.game_rewards.get_mean()
                    mean_lengths = self.game_lengths.get_mean()
                    self.mean_rewards = mean_rewards[0]

                    for i in range(self.value_size):
                        rewards_name = 'rewards' if i == 0 else 'rewards{0}'.format(i)
                        self.writer.add_scalar(rewards_name + '/step'.format(i), mean_rewards[i], frame)
                        self.writer.add_scalar(rewards_name + '/iter'.format(i), mean_rewards[i], epoch_num)
                        self.writer.add_scalar(rewards_name + '/time'.format(i), mean_rewards[i], total_time)

                    self.writer.add_scalar('episode_lengths/step', mean_lengths, frame)
                    self.writer.add_scalar('episode_lengths/iter', mean_lengths, epoch_num)
                    self.writer.add_scalar('episode_lengths/time', mean_lengths, total_time)

                    # removed equal signs (i.e. "rew=") from the checkpoint name since it messes with hydra CLI parsing
                    checkpoint_name = self.config['name'] + '_ep_' + str(epoch_num) + '_rew_' + str(mean_rewards[0])

                    if self.save_freq > 0:
                        if (epoch_num % self.save_freq == 0) and (mean_rewards <= self.last_mean_rewards):
                            self.save(os.path.join(self.nn_dir, 'last_' + checkpoint_name))

                    if mean_rewards[0] > self.last_mean_rewards and epoch_num >= self.save_best_after:
                        print('saving next best rewards: ', mean_rewards)
                        self.last_mean_rewards = mean_rewards[0]
                        self.save(os.path.join(self.nn_dir, self.config['name']))

                        if 'score_to_win' in self.config:
                            if self.last_mean_rewards > self.config['score_to_win']:
                                print('Network won!')
                                self.save(os.path.join(self.nn_dir, checkpoint_name))
                                should_exit = True

                if epoch_num >= self.max_epochs:
                    if self.game_rewards.current_size == 0:
                        print('WARNING: Max epochs reached before any env terminated at least once')
                        mean_rewards = -np.inf

                    self.save(os.path.join(self.nn_dir,
                                           'last_' + self.config['name'] + 'ep' + str(epoch_num) + 'rew' + str(
                                               mean_rewards)))
                    print('MAX EPOCHS NUM!')
                    should_exit = True
                tot_win_rate = 0
                tot_games_num = 0
                # self_player process
                for player in self.player_pool.players:
                    win_rate = 1 - player.win_rate()
                    self.writer.add_scalar(f'rate/win_rate_player_{player.player_idx}', win_rate, epoch_num)
                    tot_win_rate += win_rate * player.games_num()
                    tot_games_num += player.games_num()
                if tot_games_num > 0:
                    self.check_op_update(tot_win_rate / tot_games_num)
                if self.epoch_num % 10 == 0:
                    self.resample_op()
                update_time = 0

            if self.multi_gpu:
                should_exit_t = torch.tensor(should_exit, device=self.device).float()
                dist.broadcast(should_exit_t, 0)
                should_exit = should_exit_t.bool().item()
            if should_exit:
                return self.last_mean_rewards, epoch_num

    def get_action_values(self, obs, is_op=False):
        processed_obs = self._preproc_obs(obs['obs_op'] if is_op else obs['obs'])
        self.model.eval()
        input_dict = {
            'is_train': False,
            'prev_actions': None,
            'obs': processed_obs,
            'rnn_states': self.rnn_states
        }
        with torch.no_grad():
            if is_op:
                res_dict = {
                    "actions": torch.zeros((self.num_actors, self.actions_num), device=self.device),
                    "values": torch.zeros((self.num_actors, 1), device=self.device)
                }
                self.player_pool.inference(input_dict, res_dict, processed_obs)
                self_env_indices = torch.arange(end=self.fixed_env_num, device=self.device, dtype=torch.long)
                input_dict['obs'] = processed_obs[self_env_indices]
                out_dict = self.model(input_dict)
                for key in res_dict:
                    res_dict[key][self_env_indices] = out_dict[key]
            else:
                res_dict = self.model(input_dict)
            if self.has_central_value:
                states = obs['states']
                input_dict = {
                    'is_train': False,
                    'states': states,
                }
                value = self.get_central_value(input_dict)
                res_dict['values'] = value
        return res_dict

    def resample_op(self):
        env_indices = torch.arange(start=self.fixed_env_num, end=self.num_actors, device=self.device, dtype=torch.long,
                                   requires_grad=False)
        step = (self.num_actors - self.fixed_env_num) // self.op_batch_num
        for player in self.player_pool.players:
            player.clear_envs()
        for i in range(0, self.num_actors - self.fixed_env_num, step):
            player = self.player_pool.sample_player()
            player.add_envs(env_indices[i:i + step])
        print("resample done")

    def restore_op(self, fn):
        checkpoint = torch_ext.load_checkpoint(fn)
        self.op_model.load_state_dict(checkpoint['model'])
        if self.normalize_input and 'running_mean_std' in checkpoint:
            self.op_model.running_mean_std.load_state_dict(checkpoint['running_mean_std'])

    def check_op_update(self, win_rate):
        done_count = self.vec_env.env.win_count + self.vec_env.env.lose_count + self.vec_env.env.draw_count + 1
        # print(win_rate, done_count)
        if done_count > self.games_to_check and win_rate > self.update_win_rate:
            if self.update_win_rate > 0.6:
                self.update_win_rate -= 0.03
            print(f'winrate:{win_rate},add opponent to player pool')
            self.update_op_num += 1
            self.update_player_pool(self.model, player_idx=self.update_op_num)
            self.vec_env.env.clear_count()
            self.save(os.path.join(self.nn_dir, f'policy_{self.update_op_num}'))
        self.writer.add_scalar('rate/win_rate', win_rate, self.epoch_num)

    def create_model(self):
        model = self.network.build(self.base_model_config)
        model.to(self.device)
        return model

    def update_player_pool(self, model, player_idx):
        new_model = self.create_model()
        new_model.load_state_dict(copy.deepcopy(model.state_dict()))
        if hasattr(model, 'running_mean_std'):
            new_model.running_mean_std.load_state_dict(copy.deepcopy(model.running_mean_std.state_dict()))
        player = SinglePlayer(player_idx, new_model, self.device)
        self.player_pool.add_player(player)
        self.resample_op()
