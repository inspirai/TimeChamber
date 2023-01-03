# License: see [LICENSE, LICENSES/isaacgymenvs/LICENSE]
import os
import time
import torch
import numpy as np
from rl_games.algos_torch import players
import random
from rl_games.algos_torch import torch_ext
from rl_games.common.tr_helpers import unsqueeze_obs
from timechamber.ase import hrl_players
from timechamber.utils.utils import load_check, load_checkpoint
from .pfsp_player_pool import PFSPPlayerPool, PFSPPlayerVectorizedPool, PFSPPlayerThreadPool, PFSPPlayerProcessPool, \
    SinglePlayer
import matplotlib.pyplot as plt

from multielo import MultiElo


class HRLSPPlayer(hrl_players.HRLPlayer):
    def __init__(self, params):
        params['config']['device_name'] = params['device']
        super().__init__(params)
        print(f'params:{params}')
        self.network = self.config['network']
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
        self.record_elo = self.player_config.get('record_elo', False)
        self.init_elo = self.player_config.get('init_elo', 400)
        self.num_actors = params['config']['num_actors']
        self.player_pool_type = params['player_pool_type']
        self.player_pool = None
        self.op_player_pool = None
        self.num_opponents = params['num_agents'] - 1
        self.max_steps = 1000
        self.update_op_num = 0
        self.players_per_env = []
        self.elo = MultiElo()

    def restore(self, load_dir):
        if os.path.isdir(load_dir):
            self.player_pool = self._build_player_pool(params=self.params, player_num=len(os.listdir(load_dir)))
            print('dir:', load_dir)
            sorted_players = []
            for idx, policy_check_checkpoint in enumerate(os.listdir(load_dir)):
                model_timestep = os.path.getmtime(load_dir + '/' + str(policy_check_checkpoint))
                self.policy_timestep.append(model_timestep)
                model = self.load_model(load_dir + '/' + str(policy_check_checkpoint))
                new_player = SinglePlayer(player_idx=model_timestep, model=model, device=self.device,
                                          rating=self.init_elo, obs_batch_len=self.num_actors * self.num_opponents)
                sorted_players.append(new_player)
            sorted_players.sort(key=lambda player: player.player_idx)
            for idx, player in enumerate(sorted_players):
                player.player_idx = idx
                self.player_pool.add_player(player)
            self.policy_timestep.sort()
        else:
            self.player_pool = self._build_player_pool(params=self.params, player_num=1)
            self.policy_timestep.append(os.path.getmtime(load_dir))
            model = self.load_model(load_dir)
            new_player = SinglePlayer(player_idx=0, model=model, device=self.device,
                                      rating=self.init_elo, obs_batch_len=self.num_actors * self.num_opponents)
            self.player_pool.add_player(new_player)
        self.restore_op(self.params['op_load_path'])
        self._norm_policy_timestep()
        self._alloc_env_indices()

    def restore_op(self, load_dir):
        if os.path.isdir(load_dir):
            self.op_player_pool = self._build_player_pool(params=self.params, player_num=len(os.listdir(load_dir)))
            sorted_players = []
            for idx, policy_check_checkpoint in enumerate(os.listdir(load_dir)):
                model_timestep = os.path.getmtime(load_dir + '/' + str(policy_check_checkpoint))
                self.policy_op_timestep.append(model_timestep)
                model = self.load_model(load_dir + '/' + str(policy_check_checkpoint))
                new_player = SinglePlayer(player_idx=model_timestep, model=model, device=self.device,
                                          rating=self.init_elo, obs_batch_len=self.num_actors * self.num_opponents)
                sorted_players.append(new_player)
            sorted_players.sort(key=lambda player: player.player_idx)
            for idx, player in enumerate(sorted_players):
                player.player_idx = idx
                self.op_player_pool.add_player(player)
            self.policy_op_timestep.sort()
        else:
            self.op_player_pool = self._build_player_pool(params=self.params, player_num=1)
            self.policy_op_timestep.append(os.path.getmtime(load_dir))
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
            return PFSPPlayerProcessPool(max_length=player_num,
                                         device=self.device)
        elif self.player_pool_type == 'multi_process':
            return PFSPPlayerThreadPool(max_length=player_num,
                                        device=self.device)
        elif self.player_pool_type == 'vectorized':
            vector_model_config = self.base_model_config
            vector_model_config['num_envs'] = self.num_actors * self.num_opponents
            vector_model_config['population_size'] = player_num

            return PFSPPlayerVectorizedPool(max_length=player_num, device=self.device,
                                            vector_model_config=vector_model_config, params=params)
        else:
            return PFSPPlayerPool(max_length=player_num, device=self.device)

    def _update_rating(self, info, env_indices):
        for env_idx in env_indices:
            if self.num_opponents == 1:
                player = self.players_per_env[env_idx][0]
                op_player = self.players_per_env[env_idx][1]
                if info['win'][env_idx]:
                    player.rating, op_player.rating = self.elo.get_new_ratings([player.rating, op_player.rating])
                elif info['lose'][env_idx]:
                    op_player.rating, player.rating = self.elo.get_new_ratings([op_player.rating, player.rating])
                elif info['draw'][env_idx]:
                    player.rating, op_player.rating = self.elo.get_new_ratings([player.rating, op_player.rating],
                                                                               result_order=[1, 1])
            else:
                ranks = info['ranks'][env_idx].cpu().numpy()
                players_sorted_by_rank = sorted(enumerate(self.players_per_env[env_idx]), key=lambda x: ranks[x[0]])
                sorted_ranks = sorted(ranks)
                now_ratings = [player.rating for idx, player in players_sorted_by_rank]
                new_ratings = self.elo.get_new_ratings(now_ratings, result_order=sorted_ranks)
                for idx, new_rating in enumerate(new_ratings):
                    players_sorted_by_rank[idx][1].rating = new_rating

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
            done_indices = torch.tensor([], device=self.device, dtype=torch.long)

            for n in range(self.max_steps):
                obses = self.env_reset(self.env, done_indices)
                if has_masks:
                    masks = self.env.get_action_mask()
                    action = self.get_masked_action(
                        obses, masks, is_determenistic)
                else:
                    action = self.get_action(obses['obs'], is_determenistic)
                    action_op = self.get_action(obses['obs_op'], is_determenistic, is_op=True)
                obses, r, done, info = self.env_step(self.env, obses, action, action_op)
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
                done_indices = done_indices[:, 0]
        if self.record_elo:
            self._plot_elo_curve()

    def _plot_elo_curve(self):
        x = np.array(self.policy_timestep)
        y = np.arange(len(self.player_pool.players))
        x_op = np.array(self.policy_op_timestep)
        y_op = np.arange(len(self.op_player_pool.players))
        for player in self.player_pool.players:
            idx = player.player_idx
            y[idx] = player.rating
        for player in self.op_player_pool.players:
            idx = player.player_idx
            y_op[idx] = player.rating
        if self.params['load_path'] != self.params['op_load_path']:
            l1 = plt.plot(x, y, 'b--', label='policy')
            l2 = plt.plot(x_op, y_op, 'r--', label='policy_op')
            plt.plot(x, y, 'b^-', x_op, y_op, 'ro-')
        else:
            l1 = plt.plot(x, y, 'b--', label='policy')
            plt.plot(x, y, 'b^-')
        plt.title('ELO Curve')
        plt.xlabel('timestep/days')
        plt.ylabel('ElO')
        plt.legend()
        plt.savefig(self.params['load_path'] + '/../elo.jpg')

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
        if is_determenistic:
            current_action = mu
        else:
            current_action = action

        current_action = torch.squeeze(current_action.detach())
        return torch.clamp(current_action, -1.0, 1.0)

    def _norm_policy_timestep(self):
        self.policy_op_timestep.sort()
        self.policy_timestep.sort()
        for idx in range(1, len(self.policy_op_timestep)):
            self.policy_op_timestep[idx] -= self.policy_op_timestep[0]
            self.policy_op_timestep[idx] /= 3600 * 24
        for idx in range(1, len(self.policy_timestep)):
            self.policy_timestep[idx] -= self.policy_timestep[0]
            self.policy_timestep[idx] /= 3600 * 24
        self.policy_timestep[0] = 0
        if len(self.policy_op_timestep):
            self.policy_op_timestep[0] = 0

    def env_reset(self, env, env_ids=None):
        obs = env.reset(env_ids)
        obs_dict = {}
        obs_dict['obs_op'] = obs[self.num_actors:]
        obs_dict['obs'] = obs[:self.num_actors]
        return obs_dict

    def env_step(self, env, obs_dict, ego_actions, op_actions):
        obs = obs_dict['obs']
        obs_op = obs_dict['obs_op']
        rewards = 0.0
        done_count = 0.0
        disc_rewards = 0.0
        terminate_count = 0.0
        win_count = 0.0
        lose_count = 0.0
        draw_count = 0.0

        for t in range(self._llc_steps):
            llc_ego_actions = self._compute_llc_action(obs, ego_actions)
            llc_op_actions = self._compute_llc_action(obs_op, op_actions)
            llc_actions = torch.cat((llc_ego_actions, llc_op_actions), dim=0)
            obs_all, curr_rewards, curr_dones, infos = env.step(llc_actions)

            rewards += curr_rewards
            done_count += curr_dones

            terminate_count += infos['terminate']
            win_count += infos['win']
            lose_count += infos['lose']
            draw_count += infos['draw']

            amp_obs = infos['amp_obs']
            curr_disc_reward = self._calc_disc_reward(amp_obs)
            curr_disc_reward = curr_disc_reward[0, 0].cpu().numpy()
            disc_rewards += curr_disc_reward

            obs = obs_all[:self.num_actors]
            obs_op = obs_all[self.num_actors:]

        rewards /= self._llc_steps
        disc_rewards /= self._llc_steps
        dones = torch.zeros_like(done_count)
        dones[done_count > 0] = 1.0
        terminate = torch.zeros_like(terminate_count)
        terminate[terminate_count > 0] = 1.0
        infos['terminate'] = terminate
        infos['disc_rewards'] = disc_rewards

        wins = torch.zeros_like(win_count)
        wins[win_count > 0] = 1.0
        infos['win'] = wins
        
        loses = torch.zeros_like(lose_count)
        loses[lose_count > 0] = 1.0
        infos['lose'] = loses
        
        draws = torch.zeros_like(draw_count)
        draws[draw_count > 0] = 1.0
        infos['draw'] = draws

        next_obs_dict = {}
        next_obs_dict['obs_op'] = obs_op
        next_obs_dict['obs'] = obs

        if self.value_size > 1:
            rewards = rewards[0]
        if self.is_tensor_obses:
            return self.obs_to_torch(next_obs_dict), rewards.cpu(), dones.cpu(), infos
        else:
            if np.isscalar(dones):
                rewards = np.expand_dims(np.asarray(rewards), 0)
                dones = np.expand_dims(np.asarray(dones), 0)
            return next_obs_dict, rewards, dones, infos

    def create_model(self):
        model = self.network.build(self.base_model_config)
        model.to(self.device)
        return model

    def load_model(self, fn):
        model = self.create_model()
        checkpoint = load_checkpoint(fn, device=self.device)
        checkpoint = load_check(checkpoint, normalize_input=self.normalize_input,
                                normalize_value=self.normalize_value)

        model.load_state_dict(checkpoint['model'])

        if self.normalize_input and 'running_mean_std' in checkpoint:
            model.running_mean_std.load_state_dict(checkpoint['running_mean_std'])

        return model
