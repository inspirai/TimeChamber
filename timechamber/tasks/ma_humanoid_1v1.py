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

import torch
import os
from isaacgym import gymapi, gymtorch, gymutil
from isaacgym.torch_utils import *

from .ase_humanoid.humanoid_amp import HumanoidAMP
from .ase_humanoid.humanoid_amp_task import HumanoidAMPTask
from timechamber.utils import torch_jit_utils as torch_utils
from typing import Dict, Any, Tuple


def parse_sim_params(physics_engine: str, config_sim: Dict[str, Any]) -> gymapi.SimParams:
    """Parse the config dictionary for physics stepping settings.

    Args:
        physics_engine: which physics engine to use. "physx" or "flex"
        config_sim: dict of sim configuration parameters
    Returns
        IsaacGym SimParams object with updated settings.
    """
    sim_params = gymapi.SimParams()

    # check correct up-axis
    if config_sim["up_axis"] not in ["z", "y"]:
        msg = f"Invalid physics up-axis: {config_sim['up_axis']}"
        print(msg)
        raise ValueError(msg)

    # assign general sim parameters
    sim_params.dt = config_sim["dt"]
    sim_params.num_client_threads = config_sim.get("num_client_threads", 0)
    sim_params.use_gpu_pipeline = config_sim["use_gpu_pipeline"]
    sim_params.substeps = config_sim.get("substeps", 2)

    # assign up-axis
    if config_sim["up_axis"] == "z":
        sim_params.up_axis = gymapi.UP_AXIS_Z
    else:
        sim_params.up_axis = gymapi.UP_AXIS_Y

    # assign gravity
    sim_params.gravity = gymapi.Vec3(*config_sim["gravity"])

    # configure physics parameters
    if physics_engine == "physx":
        # set the parameters
        if "physx" in config_sim:
            for opt in config_sim["physx"].keys():
                if opt == "contact_collection":
                    setattr(sim_params.physx, opt, gymapi.ContactCollection(config_sim["physx"][opt]))
                else:
                    setattr(sim_params.physx, opt, config_sim["physx"][opt])
    else:
        # set the parameters
        if "flex" in config_sim:
            for opt in config_sim["flex"].keys():
                setattr(sim_params.flex, opt, config_sim["flex"][opt])

    # return the configured params
    return sim_params


class MA_Humanoid_1v1(HumanoidAMPTask):
    def __init__(self, cfg, sim_device, rl_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        split_device = sim_device.split(":")
        device_type = split_device[0]
        device_id = int(split_device[1]) if len(split_device) > 1 else 0
        cfg['sim_device'] = sim_device
        cfg['rl_device'] = rl_device
        cfg['headless'] = headless
        super().__init__(cfg=cfg,
                         sim_params=parse_sim_params(cfg["physics_engine"], cfg["sim"]),
                         physics_engine=gymapi.SIM_PHYSX,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)

        self._tar_dist_min = 0.5
        self._tar_dist_max = 10.0
        self._near_dist = 1.5
        self._near_prob = 0.5

        self._prev_root_pos = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)

        strike_body_names = cfg["env"]["strikeBodyNames"]
        self._strike_body_ids = self._build_strike_body_ids_tensor(self.envs[0], self.humanoid_handles[0],
                                                                   strike_body_names)
        self._build_target_tensors()
        self.reset_idx()

        return

    def get_task_obs_size(self):
        obs_size = 0
        if (self._enable_task_obs):
            obs_size = 15
        return obs_size

    def _create_envs(self, num_envs, spacing, num_per_row):
        self._target_handles = []
        self._load_target_asset()

        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')
        asset_file = "mjcf/amp_humanoid_sword_shield.xml"

        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        # asset_options.fix_base_link = True
        print('test', asset_root, asset_file)
        humanoid_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        actuator_props = self.gym.get_asset_actuator_properties(humanoid_asset)
        motor_efforts = [prop.motor_effort for prop in actuator_props]

        # create force sensors at the feet
        right_foot_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, "right_foot")
        left_foot_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, "left_foot")
        sensor_pose = gymapi.Transform()

        self.gym.create_asset_force_sensor(humanoid_asset, right_foot_idx, sensor_pose)
        self.gym.create_asset_force_sensor(humanoid_asset, left_foot_idx, sensor_pose)

        self.max_motor_effort = max(motor_efforts)
        self.motor_efforts = to_torch(motor_efforts, device=self.device)

        self.torso_index = 0
        self.num_bodies = self.gym.get_asset_rigid_body_count(humanoid_asset)
        self.num_dof = self.gym.get_asset_dof_count(humanoid_asset)
        self.num_joints = self.gym.get_asset_joint_count(humanoid_asset)

        self.humanoid_handles = []
        self.envs = []
        self.dof_limits_lower = []
        self.dof_limits_upper = []

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self._build_env(i, env_ptr, humanoid_asset)
            self.envs.append(env_ptr)

        dof_prop = self.gym.get_actor_dof_properties(self.envs[0], self.humanoid_handles[0])
        for j in range(self.num_dof):
            if dof_prop['lower'][j] > dof_prop['upper'][j]:
                self.dof_limits_lower.append(dof_prop['upper'][j])
                self.dof_limits_upper.append(dof_prop['lower'][j])
            else:
                self.dof_limits_lower.append(dof_prop['lower'][j])
                self.dof_limits_upper.append(dof_prop['upper'][j])

        self.dof_limits_lower = to_torch(self.dof_limits_lower, device=self.device)
        self.dof_limits_upper = to_torch(self.dof_limits_upper, device=self.device)

        if (self._pd_control):
            self._build_pd_action_offset_scale()

        return

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        super()._build_env(env_id, env_ptr, humanoid_asset)
        self._build_target(env_id, env_ptr)
        return

    def _load_target_asset(self):
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')
        asset_file = "mjcf/strike_target.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.linear_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.density = 30.0
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        self._target_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        return

    def _build_target(self, env_id, env_ptr):
        col_group = env_id
        col_filter = 0
        segmentation_id = 0

        default_pose = gymapi.Transform()
        default_pose.p.x = 1.0

        target_handle = self.gym.create_actor(env_ptr, self._target_asset, default_pose, "target", col_group,
                                              col_filter, segmentation_id)
        self._target_handles.append(target_handle)

        return

    def _build_strike_body_ids_tensor(self, env_ptr, actor_handle, body_names):
        env_ptr = self.envs[0]
        actor_handle = self.humanoid_handles[0]
        body_ids = []

        for body_name in body_names:
            body_id = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, body_name)
            assert (body_id != -1)
            body_ids.append(body_id)

        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids

    def _build_target_tensors(self):
        num_actors = self.get_num_actors_per_env()
        self._target_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., 1, :]

        self._tar_actor_ids = to_torch(num_actors * np.arange(self.num_envs), device=self.device, dtype=torch.int32) + 1

        bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor)
        self._tar_contact_forces = contact_force_tensor.view(self.num_envs, bodies_per_env, 3)[..., self.num_bodies, :]

        return

    def _reset_actors(self, env_ids):
        super()._reset_actors(env_ids)
        self._reset_target(env_ids)
        return

    def _reset_target(self, env_ids):
        n = len(env_ids)

        init_near = torch.rand([n], dtype=self._target_states.dtype,
                               device=self._target_states.device) < self._near_prob
        dist_max = self._tar_dist_max * torch.ones([n], dtype=self._target_states.dtype,
                                                   device=self._target_states.device)
        dist_max[init_near] = self._near_dist
        rand_dist = (dist_max - self._tar_dist_min) * torch.rand([n], dtype=self._target_states.dtype,
                                                                 device=self._target_states.device) + self._tar_dist_min

        rand_theta = 2 * np.pi * torch.rand([n], dtype=self._target_states.dtype, device=self._target_states.device)
        self._target_states[env_ids, 0] = rand_dist * torch.cos(rand_theta) + self._humanoid_root_states[env_ids, 0]
        self._target_states[env_ids, 1] = rand_dist * torch.sin(rand_theta) + self._humanoid_root_states[env_ids, 1]
        self._target_states[env_ids, 2] = 0.9

        rand_rot_theta = 2 * np.pi * torch.rand([n], dtype=self._target_states.dtype, device=self._target_states.device)
        axis = torch.tensor([0.0, 0.0, 1.0], dtype=self._target_states.dtype, device=self._target_states.device)
        rand_rot = quat_from_angle_axis(rand_rot_theta, axis)

        self._target_states[env_ids, 3:7] = rand_rot
        self._target_states[env_ids, 7:10] = 0.0
        self._target_states[env_ids, 10:13] = 0.0
        return

    def _reset_env_tensors(self, env_ids):
        super()._reset_env_tensors(env_ids)

        env_ids_int32 = self._tar_actor_ids[env_ids]
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        return

    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
        self._prev_root_pos[:] = self._humanoid_root_states[..., 0:3]
        return

    def _compute_task_obs(self, env_ids=None):
        if (env_ids is None):
            root_states = self._humanoid_root_states
            tar_states = self._target_states
        else:
            root_states = self._humanoid_root_states[env_ids]
            tar_states = self._target_states[env_ids]

        obs = compute_battle_observations(root_states, tar_states)
        return obs

    def _compute_reward(self, actions):
        tar_pos = self._target_states[..., 0:3]
        tar_rot = self._target_states[..., 3:7]
        char_root_state = self._humanoid_root_states
        strike_body_vel = self._rigid_body_vel[..., self._strike_body_ids[0], :]

        self.rew_buf[:] = compute_battle_reward(tar_pos, tar_rot, char_root_state,
                                                self._prev_root_pos, strike_body_vel,
                                                self.dt, self._near_dist)
        return

    def _compute_reset(self):
        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(self.reset_buf, self.progress_buf,
                                                                           self._contact_forces, self._contact_body_ids,
                                                                           self._rigid_body_pos,
                                                                           self._tar_contact_forces,
                                                                           self._strike_body_ids,
                                                                           self.max_episode_length,
                                                                           self._enable_early_termination,
                                                                           self._termination_heights)
        return

    def _draw_task(self):
        cols = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)

        self.gym.clear_lines(self.viewer)

        starts = self._humanoid_root_states[..., 0:3]
        ends = self._target_states[..., 0:3]
        verts = torch.cat([starts, ends], dim=-1).cpu().numpy()

        for i, env_ptr in enumerate(self.envs):
            curr_verts = verts[i]
            curr_verts = curr_verts.reshape([1, 6])
            self.gym.add_lines(self.viewer, env_ptr, curr_verts.shape[0], curr_verts, cols)

        return


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
        obs_buf_op,
        reset_buf,
        progress_buf,
        pos_before,
        torques,
        hp,
        hp_op,
        termination_height,
        max_episode_length,
        borderline_space,
        draw_penalty_scale,
        win_reward_scale,
        move_to_op_reward_scale,
        stay_in_center_reward_scale,
        action_cost_scale,
        push_scale,
        joints_at_limit_cost_scale,
        dense_reward_scale,
        hp_decay_scale,
        dt,
):
    # type: (Tensor, Tensor, Tensor, Tensor,Tensor,Tensor,Tensor,Tensor,float, float,float, float,float,float,float,float,float,float,float,float,float) -> Tuple[Tensor, Tensor,Tensor,Tensor,Tensor,Tensor,Tensor]

    hp -= (obs_buf[:, 2] < termination_height) * hp_decay_scale
    hp_op -= (obs_buf_op[:, 2] < termination_height) * hp_decay_scale
    # print(hp, hp_op)
    is_out = torch.sum(torch.square(obs_buf[:, 0:2]), dim=-1) >= borderline_space ** 2
    is_out_op = torch.sum(torch.square(obs_buf_op[:, 0:2]), dim=-1) >= borderline_space ** 2
    #
    is_out = is_out | (hp <= 0)
    is_out_op = is_out_op | (hp_op <= 0)
    # reset agents
    tmp_ones = torch.ones_like(reset_buf)
    reset = torch.where(is_out, tmp_ones, reset_buf)
    reset = torch.where(is_out_op, tmp_ones, reset)
    reset = torch.where(progress_buf >= max_episode_length - 1, tmp_ones, reset)

    hp = torch.where(reset > 0, tmp_ones * 100., hp)
    hp_op = torch.where(reset > 0, tmp_ones * 100., hp_op)

    win_reward = win_reward_scale * is_out_op
    lose_penalty = -win_reward_scale * is_out
    draw_penalty = torch.where(progress_buf >= max_episode_length - 1, tmp_ones * draw_penalty_scale,
                               torch.zeros_like(reset, dtype=torch.float))
    move_reward = compute_move_reward(obs_buf[:, 0:2], pos_before,
                                      obs_buf_op[:, 0:2], dt,
                                      move_to_op_reward_scale)
    # stay_in_center_reward = stay_in_center_reward_scale * torch.exp(-torch.linalg.norm(obs_buf[:, :2], dim=-1))
    dof_at_limit_cost = torch.sum(obs_buf[:, 13:21] > 0.99, dim=-1) * joints_at_limit_cost_scale
    push_reward = -push_scale * torch.exp(-torch.linalg.norm(obs_buf_op[:, :2], dim=-1))
    action_cost_penalty = torch.sum(torch.square(torques), dim=1) * action_cost_scale
    # print("torques:", torques[0, 2])
    not_move_penalty = -10 * torch.exp(-torch.sum(torch.abs(torques), dim=1))
    # print(f'action:...{action_cost_penalty.shape}')
    dense_reward = move_reward + dof_at_limit_cost + push_reward + action_cost_penalty + not_move_penalty
    total_reward = win_reward + lose_penalty + draw_penalty + dense_reward * dense_reward_scale

    return total_reward, reset, hp, hp_op, is_out_op, is_out, progress_buf >= max_episode_length - 1


@torch.jit.script
def compute_ant_observations(
        root_states,
        root_states_op,
        dof_pos,
        dof_vel,
        dof_limits_lower,
        dof_limits_upper,
        dof_vel_scale,
        termination_height
):
    # type: (Tensor,Tensor,Tensor,Tensor,Tensor,Tensor,float,float)->Tensor
    dof_pos_scaled = unscale(dof_pos, dof_limits_lower, dof_limits_upper)
    # print("dof pos:", dof_pos[0, :])
    obs = torch.cat(
        (root_states[:, :13], dof_pos_scaled, dof_vel * dof_vel_scale, root_states_op[:, :7],
         root_states[:, :2] - root_states_op[:, :2], torch.unsqueeze(root_states[:, 2] < termination_height, -1),
         torch.unsqueeze(root_states_op[:, 2] < termination_height, -1)), dim=-1)

    return obs
