# Copyright (c) 2018-2021, NVIDIA Corporation
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

from typing import Tuple
import os

import torch
from isaacgym import gymtorch
from isaacgym.gymtorch import *

from isaacgymenvs.utils.torch_jit_utils import *
from .base.ma_vec_task import MA_VecTask


class BattleAgent:
    def __init__(self, agent_idx, root_state, dof_pos, dof_vel):
        self.agent_idx = agent_idx
        self.root_state = root_state
        self.dof_pos = dof_pos
        self.dof_vel = dof_vel


class MA_Ant_Battle(MA_VecTask):

    def __init__(self, cfg, sim_device, rl_device, graphics_device_id, headless, virtual_screen_capture, force_render):

        self.extras = None
        self.cfg = cfg
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.randomize = self.cfg["task"]["randomize"]

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.termination_height = self.cfg["env"]["terminationHeight"]
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]
        self.action_scale = self.cfg["env"]["control"]["actionScale"]
        self.joints_at_limit_cost_scale = self.cfg["env"]["jointsAtLimitCost"]
        self.dof_vel_scale = self.cfg["env"]["dofVelocityScale"]
        self.ant_agents_state = []
        self.win_reward_scale = 2000
        self.move_to_op_reward_scale = 1.
        self.stay_in_center_reward_scale = 0.2
        self.action_cost_scale = -0.000025
        self.push_scale = 1.
        self.dense_reward_scale = 1.0
        self.hp_decay_scale = 1.
        self.Kp = self.cfg["env"]["control"]["stiffness"]
        self.Kd = self.cfg["env"]["control"]["damping"]
        self.cfg["env"]["numObservations"] = 32 + 27 * (self.cfg["env"].get("numAgents", 1) - 1)
        self.cfg["env"]["numActions"] = 8
        self.borderline_space = cfg["env"]["borderlineSpace"]
        self.borderline_space_unit = self.borderline_space / self.max_episode_length
        self.ant_body_colors = [gymapi.Vec3(0.97, 0.38, 0.06), gymapi.Vec3(0.24, 0.38, 0.06),
                                gymapi.Vec3(0.56, 0.85, 0.25), gymapi.Vec3(0.44, 0.38, 0.68),
                                gymapi.Vec3(0.14, 0.97, 0.24), gymapi.Vec3(0.63, 0.2, 0.87),
                                gymapi.Vec3(0.52, 0.13, 0.23), gymapi.Vec3(0.26, 0.78, 0.98), ]
        super().__init__(config=self.cfg, sim_device=sim_device, rl_device=rl_device,
                         graphics_device_id=graphics_device_id,
                         headless=headless)

        self.use_central_value = False
        self.obs_idxs = torch.eye(4, dtype=torch.float32, device=self.device)
        if self.viewer is not None:
            for i, env in enumerate(self.envs):
                self._add_circle_borderline(env, self.borderline_space)
            cam_pos = gymapi.Vec3(15.0, 0.0, 3.4)
            cam_target = gymapi.Vec3(10.0, 0.0, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)

        sensors_per_env = 4
        self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs,
                                                                          sensors_per_env * 6)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)

        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        print(f'root_states:{self.root_states.shape}')
        self.initial_root_states = self.root_states.clone()
        self.initial_root_states[:, 7:13] = 0  # set lin_vel and ang_vel to 0

        # create some wrapper tensors for different slices
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        print(f'dof:{self.dof_state.shape}')
        dof_state_shaped = self.dof_state.view(self.num_envs, -1, 2)
        for idx in range(self.num_agents):
            ant_root_state = self.root_states[idx::self.num_agents]
            ant_dof_pos = dof_state_shaped[:, idx * self.num_dof:(idx + 1) * self.num_dof, 0]
            ant_dof_vel = dof_state_shaped[:, idx * self.num_dof:(idx + 1) * self.num_dof, 1]
            ant_agent = BattleAgent(idx, ant_root_state, ant_dof_pos, ant_dof_vel)
            self.ant_agents_state.append((ant_root_state, ant_dof_pos, ant_dof_vel))

        self.initial_dof_pos = torch.zeros_like(self.ant_agents_state[0][1], device=self.device, dtype=torch.float)
        zero_tensor = torch.tensor([0.0], device=self.device)
        self.initial_dof_pos = torch.where(self.dof_limits_lower > zero_tensor, self.dof_limits_lower,
                                           torch.where(self.dof_limits_upper < zero_tensor, self.dof_limits_upper,
                                                       self.initial_dof_pos))
        self.initial_dof_vel = torch.zeros_like(self.ant_agents_state[0][2], device=self.device, dtype=torch.float)
        self.dt = self.cfg["sim"]["dt"]

        torques = self.gym.acquire_dof_force_tensor(self.sim)
        self.torques = gymtorch.wrap_tensor(torques).view(self.num_envs, self.num_agents * self.num_dof)

        self.x_unit_tensor = to_torch([1, 0, 0], dtype=torch.float, device=self.device).repeat(
            (self.num_agents * self.num_envs, 1))
        self.y_unit_tensor = to_torch([0, 1, 0], dtype=torch.float, device=self.device).repeat(
            (self.num_agents * self.num_envs, 1))
        self.z_unit_tensor = to_torch([0, 0, 1], dtype=torch.float, device=self.device).repeat(
            (self.num_agents * self.num_envs, 1))

    def allocate_buffers(self):
        self.obs_buf = torch.zeros((self.num_agents, self.num_envs, self.num_obs), device=self.device,
                                   dtype=torch.float)
        self.out_buf = torch.zeros((self.num_agents, self.num_envs))
        self.rew_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.timeout_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)
        self.progress_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)
        self.randomize_buf = torch.zeros(
            self.num_envs * self.num_agents, device=self.device, dtype=torch.long)
        self.rank_buf = torch.zeros((self.num_envs, self.num_agents), device=self.device, dtype=torch.long)
        self.extras = {'ranks': torch.zeros((self.num_agents, self.num_agents), device=self.device, dtype=torch.float)}

    def create_sim(self):
        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, 'z')
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        lines = []
        borderline_height = 0.01
        for height in range(20):
            for angle in range(360):
                begin_point = [np.cos(np.radians(angle)), np.sin(np.radians(angle)), borderline_height * height]
                end_point = [np.cos(np.radians(angle + 1)), np.sin(np.radians(angle + 1)), borderline_height * height]
                lines.append(begin_point)
                lines.append(end_point)
        self.lines = np.array(lines, dtype=np.float32)
        self._create_ground_plane()
        # self._create_height_field()
        print(f'num envs {self.num_envs} env spacing {self.cfg["env"]["envSpacing"]}')
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

        # If randomizing, apply once immediately on startup before the fist sim step
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

    def _add_borderline(self, env):
        lines = np.array(
            [[-self.borderline_space, -self.borderline_space, 0], [-self.borderline_space, self.borderline_space, 0],
             [-self.borderline_space, -self.borderline_space, 0], [self.borderline_space, -self.borderline_space, 0],
             [self.borderline_space, self.borderline_space, 0], [self.borderline_space, -self.borderline_space, 0],
             [self.borderline_space, self.borderline_space, 0], [-self.borderline_space, self.borderline_space, 0]],
            dtype=np.float32)
        colors = np.array([[1, 0, 0]] * 4, dtype=np.float32)
        # print(f'border_lines:{lines.size},{colors.size}')
        self.gym.add_lines(self.viewer, env, 4, lines, colors)

    def _add_circle_borderline(self, env, radius):
        lines = self.lines * radius
        colors = np.array([[1, 0, 0]] * (len(lines) // 2), dtype=np.float32)
        # print(f'border_lines:{lines.size},{colors.size}')
        self.gym.add_lines(self.viewer, env, len(lines) // 2, lines, colors)

    def _create_height_field(self):
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = 1.
        hf_params.row_scale = 1.
        hf_params.vertical_scale = 1.
        hf_params.nbRows = 1
        hf_params.nbColumns = 1
        hf_params.transform.p.x = -0
        hf_params.transform.p.y = -0
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.plane_static_friction
        hf_params.dynamic_friction = self.plane_dynamic_friction
        hf_params.restitution = self.plane_restitution

        self.gym.add_heightfield(self.sim, np.zeros((1), dtype=np.int16), hf_params)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')
        asset_file = "mjcf/nv_ant.xml"

        if "asset" in self.cfg["env"]:
            asset_file = self.cfg["env"]["asset"].get("assetFileName", asset_file)

        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        # Note - DOF mode is set in the MJCF file and loaded by Isaac Gym
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.angular_damping = 0.0
        ant_assets = []
        for _ in range(self.num_agents):
            ant_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
            ant_assets.append(ant_asset)
        dof_props = self.gym.get_asset_dof_properties(ant_assets[0])

        self.num_dof = self.gym.get_asset_dof_count(ant_assets[0])
        self.num_bodies = self.gym.get_asset_rigid_body_count(ant_assets[0])
        for i in range(self.num_dof):
            dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            dof_props['stiffness'][i] = self.Kp
            dof_props['damping'][i] = self.Kd

        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(-self.borderline_space + 1, -self.borderline_space + 1, 1.)
        self.start_rotation = torch.tensor([start_pose.r.x, start_pose.r.y, start_pose.r.z, start_pose.r.w],
                                           device=self.device)

        self.torso_index = 0
        self.num_bodies = self.gym.get_asset_rigid_body_count(ant_assets[0])
        body_names = [self.gym.get_asset_rigid_body_name(ant_assets[0], i) for i in range(self.num_bodies)]
        extremity_names = [s for s in body_names if "foot" in s]
        self.extremities_index = torch.zeros(len(extremity_names), dtype=torch.long, device=self.device)
        print(body_names, extremity_names, self.extremities_index)
        # create force sensors attached to the "feet"
        extremity_indices = [self.gym.find_asset_rigid_body_index(ant_assets[0], name) for name in extremity_names]
        sensor_pose = gymapi.Transform()
        for body_idx in extremity_indices:
            self.gym.create_asset_force_sensor(ant_assets[0], body_idx, sensor_pose)

        self.ant_handles = []
        self.actor_indices = []
        self.envs = []
        self.pos_before = torch.zeros(2, device=self.device)
        self.dof_limits_lower = []
        self.dof_limits_upper = []

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )
            for j in range(self.num_agents):
                ant_handle = self.gym.create_actor(env_ptr, ant_assets[j], start_pose, "ant_" + str(j), i, -1, 0)
                actor_index = self.gym.get_actor_index(env_ptr, ant_handle, gymapi.DOMAIN_SIM)
                self.gym.set_actor_dof_properties(env_ptr, ant_handle, dof_props)
                self.actor_indices.append(actor_index)
                self.gym.enable_actor_dof_force_sensors(env_ptr, ant_handle)
                # forces = self.gym.get_env_rigid_contact_forces(env_ptr)
                # num_sensors = self.gym.get_actor_force_sensor_count(env_ptr, ant_handle)
                # print(f'num_sensors:{num_sensors}')
                # print(forces)
                self.ant_handles.append(ant_handle)
                for k in range(self.num_bodies):
                    self.gym.set_rigid_body_color(
                        env_ptr, ant_handle, k, gymapi.MESH_VISUAL, self.ant_body_colors[j])
            self.envs.append(env_ptr)

        dof_prop = self.gym.get_actor_dof_properties(self.envs[0], self.ant_handles[0])

        for j in range(self.num_dof):
            if dof_prop['lower'][j] > dof_prop['upper'][j]:
                self.dof_limits_lower.append(dof_prop['upper'][j])
                self.dof_limits_upper.append(dof_prop['lower'][j])
            else:
                self.dof_limits_lower.append(dof_prop['lower'][j])
                self.dof_limits_upper.append(dof_prop['upper'][j])

        self.dof_limits_lower = to_torch(self.dof_limits_lower, device=self.device)
        self.dof_limits_upper = to_torch(self.dof_limits_upper, device=self.device)
        self.actor_indices = to_torch(self.actor_indices, device=self.device).to(dtype=torch.int32)

        for i in range(len(extremity_names)):
            self.extremities_index[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.ant_handles[0],
                                                                              extremity_names[i])

    def compute_reward(self, actions):

        self.rew_buf[:], self.reset_buf[:], self.rank_buf[:] = compute_ant_reward(
            self.obs_buf,
            self.reset_buf,
            self.progress_buf,
            self.pos_before,
            self.torques,
            self.rank_buf,
            self.termination_height,
            self.max_episode_length,
            self.borderline_space,
            self.borderline_space_unit,
            self.win_reward_scale,
            self.stay_in_center_reward_scale,
            self.action_cost_scale,
            self.push_scale,
            self.joints_at_limit_cost_scale,
            self.dense_reward_scale,
            self.dt,
            self.num_agents
        )

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        for agent_idx in range(self.num_agents):
            self.obs_buf[agent_idx, :] = compute_ant_observations(
                self.ant_agents_state,
                self.progress_buf,
                self.dof_limits_lower,
                self.dof_limits_upper,
                self.dof_vel_scale,
                self.termination_height,
                self.borderline_space_unit,
                self.borderline_space,
                self.num_agents,
                agent_idx,
            )

    def reset_idx(self, env_ids):
        # print('reset.....', env_ids)
        # Randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        positions = torch_rand_float(-0.2, 0.2, (len(env_ids), self.num_dof), device=self.device)
        velocities = torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dof), device=self.device)

        for agent_idx in range(self.num_agents):
            root_state, dof_pos, dof_vel = self.ant_agents_state[agent_idx]
            dof_pos[env_ids] = tensor_clamp(self.initial_dof_pos[env_ids] + positions, self.dof_limits_lower,
                                            self.dof_limits_upper)
            dof_vel[env_ids] = velocities
        agent_env_ids = expand_env_ids(env_ids, self.num_agents)
        env_ids_int32 = self.actor_indices[agent_env_ids]
        rand_angle = torch.rand((len(env_ids),), device=self.device) * torch.pi * 2  # generate angle in 0-360

        rand_pos = torch.ones((len(agent_env_ids), 2), device=self.device) * (
                self.borderline_space * torch.ones((len(agent_env_ids), 2), device=self.device) - torch.rand(
            (len(agent_env_ids), 2), device=self.device) * 2)

        unit_angle = 2 * torch.pi / self.num_agents
        for agent_idx in range(self.num_agents):
            rand_pos[agent_idx::self.num_agents, 0] *= torch.cos(rand_angle + agent_idx * unit_angle)
            rand_pos[agent_idx::self.num_agents, 1] *= torch.sin(rand_angle + agent_idx * unit_angle)
        rand_floats = torch_rand_float(-1.0, 1.0, (len(agent_env_ids), 1), device=self.device)
        rand_rotation = quat_from_angle_axis(rand_floats[:, 0] * np.pi, self.z_unit_tensor[agent_env_ids])
        self.root_states[agent_env_ids] = self.initial_root_states[agent_env_ids]
        self.root_states[agent_env_ids, :2] = rand_pos
        self.root_states[agent_env_ids, 3:7] = rand_rotation
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.pos_before = self.root_states[0::self.num_agents, :2].clone()
        self._record_rank(env_ids)
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.rank_buf[env_ids, :] = 0

    def _record_rank(self, env_ids):
        for agent_idx in range(self.num_agents):
            for rank in range(self.num_agents + 1):
                self.extras['ranks'][agent_idx, rank] += torch.sum(self.rank_buf[env_ids][agent_idx] == rank)
        print(self.extras)

    def pre_physics_step(self, actions):
        # actions.shape = [num_envs * num_agents, num_actions], stacked as followed:
        # {[(agent1_act_1, agent1_act2)|(agent2_act1, agent2_act2)|...]_(env0),
        #  [(agent1_act_1, agent1_act2)|(agent2_act1, agent2_act2)|...]_(env1),
        #  ... }

        self.actions = actions.clone().to(self.device).view(self.num_envs, self.num_actions * self.num_agents)
        tmp_actions = self.rank_buf.unsqueeze(-1).repeat_interleave(self.num_actions, dim=-1).view(self.num_envs,
                                                                                                   self.num_actions * self.num_agents)
        zero_actions = torch.zeros_like(tmp_actions, dtype=torch.float)
        self.actions = torch.where(tmp_actions > 0, zero_actions, self.actions)
        # self.actions = torch.cat((self.actions[:self.num_envs], self.actions[self.num_envs:]), dim=-1)

        # reshape [num_envs * num_agents, num_actions] to [num_envs, num_agents * num_actions] print(f'action_size{
        # self.actions.shape}') self.actions = self.actions.reshape(self.num_envs, self.num_agents *
        # self.num_actions) print(self.actions) targets = self.actions + self.initial_dof_pos.repeat_interleave(
        # self.num_agents, dim=0).repeat_interleave(2, dim=1)

        targets = self.actions
        # print(self.obs_buf_op)

        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(targets))
        # self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(forces))

    def post_physics_step(self):
        self.progress_buf += 1
        self.randomize_buf += 1

        resets = self.reset_buf.reshape(self.num_envs, 1).sum(dim=1)
        # print(resets)
        env_ids = (resets == 1).nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        # print(self.obs_buf)
        # print(self.obs_buf_op)
        self.compute_reward(self.actions)
        self.pos_before = self.obs_buf[0, :, :2].clone()
        if self.viewer is not None:
            self.gym.clear_lines(self.viewer)
            for i, env in enumerate(self.envs):
                self._add_circle_borderline(env, self.borderline_space - self.borderline_space_unit * self.progress_buf[
                    i].item())
        print(self.rank_buf)

    def get_number_of_agents(self):
        return self.num_agents

    def zero_actions(self) -> torch.Tensor:
        """Returns a buffer with zero actions.

        Returns:
            A buffer of zero torch actions
        """
        actions = torch.zeros([self.num_envs * self.num_agents, self.num_actions], dtype=torch.float32,
                              device=self.rl_device)

        return actions

    def clear_count(self):
        self.dense_reward_scale *= 0.9


#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def expand_env_ids(env_ids, n_agents):
    # type: (Tensor, int) -> Tensor
    device = env_ids.device
    # print(f'nanget:{n_agents}')
    agent_env_ids = torch.zeros((n_agents * len(env_ids)), device=device, dtype=torch.long)
    for idx in range(n_agents):
        agent_env_ids[idx::n_agents] = env_ids * n_agents + idx
    return agent_env_ids


@torch.jit.script
def compute_move_reward(
        pos,
        pos_before,
        target,
        dt,
        move_to_op_reward_scale
):
    # type: (Tensor,Tensor,Tensor,float,float) -> Tensor
    move_vec = (pos - pos_before) / dt
    direction = target - pos_before
    direction = torch.div(direction, torch.linalg.norm(direction, dim=-1).view(-1, 1))
    s = torch.sum(move_vec * direction, dim=-1)
    return torch.maximum(s, torch.zeros_like(s)) * move_to_op_reward_scale


@torch.jit.script
def compute_ant_reward(
        obs_buf,
        reset_buf,
        progress_buf,
        pos_before,
        torques,
        now_rank,
        termination_height,
        max_episode_length,
        borderline_space,
        borderline_space_unit,
        win_reward_scale,
        stay_in_center_reward_scale,
        action_cost_scale,
        push_scale,
        joints_at_limit_cost_scale,
        dense_reward_scale,
        dt,
        num_agents
):
    # type: (Tensor, Tensor, Tensor,Tensor,Tensor,Tensor,float,float,float,float,float,float,float,float,float,float,float,int) -> Tuple[Tensor, Tensor,Tensor]

    nxt_rank_val = num_agents - torch.count_nonzero(now_rank, dim=-1).view(-1, 1).repeat_interleave(num_agents, dim=-1)
    is_out = torch.sum(torch.square(obs_buf[:, :, 0:2]), dim=-1) >= \
             (borderline_space - progress_buf * borderline_space_unit).square()
    nxt_rank = torch.where((torch.transpose(is_out, 0, 1) > 0) & (now_rank == 0), nxt_rank_val, now_rank)
    # reset agents
    tmp_ones = torch.ones_like(reset_buf)
    reset = torch.where(is_out[0, :], tmp_ones, reset_buf)
    reset = torch.where(progress_buf >= max_episode_length - 1, tmp_ones, reset)
    nxt_rank = torch.where(reset.view(-1, 1).repeat_interleave(num_agents, dim=-1) & (nxt_rank == 0), nxt_rank_val + 1,
                           nxt_rank)

    sparse_reward = 1.0 * reset
    reward_per_rank = 2 * win_reward_scale / num_agents
    sparse_reward = sparse_reward * (win_reward_scale - (nxt_rank[:, 0] - 1) * reward_per_rank)
    stay_in_center_reward = stay_in_center_reward_scale * torch.exp(-torch.linalg.norm(obs_buf[0, :, :2], dim=-1))
    dof_at_limit_cost = torch.sum(obs_buf[0, :, 13:21] > 0.99, dim=-1) * joints_at_limit_cost_scale
    action_cost_penalty = torch.sum(torch.square(torques), dim=1) * action_cost_scale
    # print("torques:", torques[0, 2])
    not_move_penalty = torch.exp(-torch.sum(torch.abs(torques), dim=1))
    # print(f'action:...{action_cost_penalty.shape}')
    dense_reward = dof_at_limit_cost + action_cost_penalty + not_move_penalty + stay_in_center_reward
    total_reward = sparse_reward + dense_reward * dense_reward_scale

    return total_reward, reset, nxt_rank


@torch.jit.script
def compute_ant_observations(
        ant_agents_state,
        progress_buf,
        dof_limits_lower,
        dof_limits_upper,
        dof_vel_scale,
        termination_height,
        borderline_space_unit,
        borderline_space,
        num_agents,
        agent_idx,
):
    # type: (List[Tuple[Tensor,Tensor,Tensor]],Tensor,Tensor,Tensor,float,float,float,float,int,int)->Tensor
    # tot length:13+8+8+1+1+(num_agents-1)*(7+2+8+8+1)
    self_root_state, self_dof_pos, self_dof_vel = ant_agents_state[agent_idx]
    dof_pos_scaled = unscale(self_dof_pos, dof_limits_lower, dof_limits_upper)
    now_border_space = (borderline_space - progress_buf * borderline_space_unit).unsqueeze(-1)
    obs = torch.cat((self_root_state[:, :13], dof_pos_scaled, self_dof_vel * dof_vel_scale,
                     now_border_space - torch.sqrt(torch.sum(self_root_state[:, :2].square(), dim=-1)).unsqueeze(-1),
                     # dis to border
                     now_border_space,
                     torch.unsqueeze(self_root_state[:, 2] < termination_height, -1)), dim=-1)
    for op_idx in range(0, num_agents):
        if op_idx == agent_idx:
            continue
        op_root_state, op_dof_pos, op_dof_vel = ant_agents_state[op_idx]
        dof_pos_scaled = unscale(op_dof_pos, dof_limits_lower, dof_limits_upper)
        obs = torch.cat((obs, op_root_state[:, :7], self_root_state[:, :2] - op_root_state[:, :2],
                         dof_pos_scaled, op_dof_vel * dof_vel_scale,
                         now_border_space - torch.sqrt(torch.sum(op_root_state[:, :2].square(), dim=-1)).unsqueeze(-1),
                         torch.unsqueeze(op_root_state[:, 2] < termination_height, -1)), dim=-1)
    return obs


@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(quat_from_angle_axis(rand0 * np.pi, x_unit_tensor),
                    quat_from_angle_axis(rand1 * np.pi, y_unit_tensor))
