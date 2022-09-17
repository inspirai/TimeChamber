from typing import Tuple
import numpy as np
import os
import math
import torch
import random

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.gymtorch import *
# from torch.tensor import Tensor

from timechamber.utils.torch_jit_utils import *
from .base.vec_task import VecTask
from .base.ma_vec_task import MA_VecTask


# todo critic_state full obs
class MA_Ant_Sumo(MA_VecTask):

    def __init__(self, cfg, sim_device, rl_device, graphics_device_id, headless, virtual_screen_capture, force_render):

        self.cfg = cfg
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.randomize = self.cfg["task"]["randomize"]

        self.max_episode_length = self.cfg["env"]["episodeLength"]

        self.termination_height = self.cfg["env"]["terminationHeight"]
        self.borderline_space = cfg["env"]["borderlineSpace"]
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]
        self.action_scale = self.cfg["env"]["control"]["actionScale"]
        self.joints_at_limit_cost_scale = self.cfg["env"]["jointsAtLimitCost"]
        self.dof_vel_scale = self.cfg["env"]["dofVelocityScale"]

        self.draw_penalty_scale = -1000
        self.win_reward_scale = 2000
        self.move_to_op_reward_scale = 1.
        self.stay_in_center_reward_scale = 0.2
        self.action_cost_scale = -0.000025
        self.push_scale = 1.
        self.dense_reward_scale = 1.
        self.hp_decay_scale = 1.

        self.Kp = self.cfg["env"]["control"]["stiffness"]
        self.Kd = self.cfg["env"]["control"]["damping"]

        # see func: compute_ant_observations() for details
        # self.cfg["env"]["numObservations"] = 48 # dof pos(2) + dof vel(2) + dof action(2) + feet force sensor(force&torque, 6)
        self.cfg["env"][
            "numObservations"] = 40
        self.cfg["env"]["numActions"] = 8
        self.cfg["env"]["numAgents"] = 2
        self.use_central_value = False

        super().__init__(config=self.cfg, sim_device=sim_device, rl_device=rl_device,
                         graphics_device_id=graphics_device_id,
                         headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        if self.viewer is not None:
            for env in self.envs:
                self._add_circle_borderline(env)
            cam_pos = gymapi.Vec3(18.0, 0.0, 5.0)
            cam_target = gymapi.Vec3(10.0, 0.0, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)

        sensors_per_env = 4
        self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs * self.num_agents,
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
        self.dof_pos = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_dof, 0]
        self.dof_pos_op = self.dof_state.view(self.num_envs, -1, 2)[:, self.num_dof:2 * self.num_dof, 0]
        self.dof_vel = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_dof, 1]
        self.dof_vel_op = self.dof_state.view(self.num_envs, -1, 2)[:, self.num_dof:2 * self.num_dof, 1]

        self.initial_dof_pos = torch.zeros_like(self.dof_pos, device=self.device, dtype=torch.float)
        zero_tensor = torch.tensor([0.0], device=self.device)
        self.initial_dof_pos = torch.where(self.dof_limits_lower > zero_tensor, self.dof_limits_lower,
                                           torch.where(self.dof_limits_upper < zero_tensor, self.dof_limits_upper,
                                                       self.initial_dof_pos))
        self.initial_dof_vel = torch.zeros_like(self.dof_vel, device=self.device, dtype=torch.float)
        self.dt = self.cfg["sim"]["dt"]

        torques = self.gym.acquire_dof_force_tensor(self.sim)
        self.torques = gymtorch.wrap_tensor(torques).view(self.num_envs, 2 * self.num_dof)

        self.x_unit_tensor = to_torch([1, 0, 0], dtype=torch.float, device=self.device).repeat((2 * self.num_envs, 1))
        self.y_unit_tensor = to_torch([0, 1, 0], dtype=torch.float, device=self.device).repeat((2 * self.num_envs, 1))
        self.z_unit_tensor = to_torch([0, 0, 1], dtype=torch.float, device=self.device).repeat((2 * self.num_envs, 1))

        self.hp = torch.ones((self.num_envs,), device=self.device, dtype=torch.float32) * 100
        self.hp_op = torch.ones((self.num_envs,), device=self.device, dtype=torch.float32) * 100

    def allocate_buffers(self):
        self.obs_buf = torch.zeros((self.num_agents * self.num_envs, self.num_obs), device=self.device,
                                   dtype=torch.float)
        self.rew_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.timeout_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)
        self.progress_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)
        self.randomize_buf = torch.zeros(
            self.num_envs * self.num_agents, device=self.device, dtype=torch.long)
        self.extras = {
            'win': torch.zeros(((self.num_agents - 1) * self.num_envs,), device=self.device, dtype=torch.bool),
            'lose': torch.zeros(((self.num_agents - 1) * self.num_envs,), device=self.device, dtype=torch.bool),
            'draw': torch.zeros(((self.num_agents - 1) * self.num_envs,), device=self.device, dtype=torch.bool)}

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
        self.lines = np.array(lines, dtype=np.float32) * self.borderline_space
        self._create_ground_plane()
        print(f'num envs {self.num_envs} env spacing {self.cfg["env"]["envSpacing"]}')
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

        # If randomizing, apply once immediately on startup before the fist sim step
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

    def _add_circle_borderline(self, env):
        colors = np.array([[1, 0, 0]] * int(len(self.lines) / 2), dtype=np.float32)
        self.gym.add_lines(self.viewer, env, int(len(self.lines) / 2), self.lines, colors)

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

        ant_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        ant_asset_op = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        dof_props = self.gym.get_asset_dof_properties(ant_asset)

        self.num_dof = self.gym.get_asset_dof_count(ant_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(ant_asset)  # 9 = 4 x 2(front&back-end legs) + 1(torso)
        for i in range(self.num_dof):
            dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            dof_props['stiffness'][i] = self.Kp
            dof_props['damping'][i] = self.Kd

        box_pose = gymapi.Transform()
        box_pose.p = gymapi.Vec3(0, 0, 0)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(-self.borderline_space + 1, -self.borderline_space + 1, 1.)
        start_pose_op = gymapi.Transform()
        start_pose_op.p = gymapi.Vec3(self.borderline_space - 1, self.borderline_space - 1, 1.)

        print(start_pose.p, start_pose_op.p)
        self.start_rotation = torch.tensor([start_pose.r.x, start_pose.r.y, start_pose.r.z, start_pose.r.w],
                                           device=self.device)

        self.torso_index = 0
        self.num_bodies = self.gym.get_asset_rigid_body_count(ant_asset)
        body_names = [self.gym.get_asset_rigid_body_name(ant_asset, i) for i in range(self.num_bodies)]
        extremity_names = [s for s in body_names if "foot" in s]
        self.extremities_index = torch.zeros(len(extremity_names), dtype=torch.long, device=self.device)

        # create force sensors attached to the "feet"
        extremity_indices = [self.gym.find_asset_rigid_body_index(ant_asset, name) for name in extremity_names]
        sensor_pose = gymapi.Transform()
        sensor_pose_op = gymapi.Transform()
        for body_idx in extremity_indices:
            self.gym.create_asset_force_sensor(ant_asset, body_idx, sensor_pose)
            self.gym.create_asset_force_sensor(ant_asset_op, body_idx, sensor_pose_op)

        self.ant_handles = []
        self.actor_indices = []
        self.actor_indices_op = []
        self.actor_handles_op = []
        self.envs = []
        self.pos_before = torch.zeros(2, device=self.device)
        self.dof_limits_lower = []
        self.dof_limits_upper = []

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )
            ant_handle = self.gym.create_actor(env_ptr, ant_asset, start_pose, "ant", i, -1, 0)
            actor_index = self.gym.get_actor_index(env_ptr, ant_handle, gymapi.DOMAIN_SIM)
            self.gym.set_actor_dof_properties(env_ptr, ant_handle, dof_props)
            self.actor_indices.append(actor_index)
            self.gym.enable_actor_dof_force_sensors(env_ptr, ant_handle)

            ant_handle_op = self.gym.create_actor(env_ptr, ant_asset_op, start_pose_op, "ant_op", i, -1, 0)
            actor_index_op = self.gym.get_actor_index(env_ptr, ant_handle_op, gymapi.DOMAIN_SIM)
            self.gym.set_actor_dof_properties(env_ptr, ant_handle_op, dof_props)

            self.actor_indices_op.append(actor_index_op)
            for j in range(self.num_bodies):
                self.gym.set_rigid_body_color(
                    env_ptr, ant_handle, j, gymapi.MESH_VISUAL, gymapi.Vec3(0.97, 0.38, 0.06))
                self.gym.set_rigid_body_color(
                    env_ptr, ant_handle_op, j, gymapi.MESH_VISUAL, gymapi.Vec3(0.24, 0.38, 0.06))

            self.envs.append(env_ptr)
            self.ant_handles.append(ant_handle)
            self.actor_handles_op.append(ant_handle_op)

        dof_prop = self.gym.get_actor_dof_properties(env_ptr, ant_handle)

        for j in range(self.num_dof):
            if dof_prop['lower'][j] > dof_prop['upper'][j]:
                self.dof_limits_lower.append(dof_prop['upper'][j])
                self.dof_limits_upper.append(dof_prop['lower'][j])
            else:
                self.dof_limits_lower.append(dof_prop['lower'][j])
                self.dof_limits_upper.append(dof_prop['upper'][j])

        self.dof_limits_lower = to_torch(self.dof_limits_lower, device=self.device)
        self.dof_limits_upper = to_torch(self.dof_limits_upper, device=self.device)
        self.actor_indices = to_torch(self.actor_indices, device=self.device)
        self.actor_indices_op = to_torch(self.actor_indices_op, device=self.device)

        for i in range(len(extremity_names)):
            self.extremities_index[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.ant_handles[0],
                                                                              extremity_names[i])

    def compute_reward(self, actions):

        self.rew_buf[:], self.reset_buf[:], self.hp[:], self.hp_op[:], \
        self.extras['win'], self.extras['lose'], self.extras['draw'] = compute_ant_reward(
            self.obs_buf[:self.num_envs],
            self.obs_buf[self.num_envs:],
            self.reset_buf,
            self.progress_buf,
            self.pos_before,
            self.torques[:, :self.num_dof],
            self.hp,
            self.hp_op,
            self.termination_height,
            self.max_episode_length,
            self.borderline_space,
            self.draw_penalty_scale,
            self.win_reward_scale,
            self.move_to_op_reward_scale,
            self.stay_in_center_reward_scale,
            self.action_cost_scale,
            self.push_scale,
            self.joints_at_limit_cost_scale,
            self.dense_reward_scale,
            self.hp_decay_scale,
            self.dt,
        )

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.obs_buf[:self.num_envs] = \
            compute_ant_observations(
                self.root_states[0::2],
                self.root_states[1::2],
                self.dof_pos,
                self.dof_vel,
                self.dof_limits_lower,
                self.dof_limits_upper,
                self.dof_vel_scale,
                self.termination_height
            )

        self.obs_buf[self.num_envs:] = compute_ant_observations(
            self.root_states[1::2],
            self.root_states[0::2],
            self.dof_pos_op,
            self.dof_vel_op,
            self.dof_limits_lower,
            self.dof_limits_upper,
            self.dof_vel_scale,
            self.termination_height
        )

    def reset_idx(self, env_ids):
        # print('reset.....', env_ids)
        # Randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        positions = torch_rand_float(-0.2, 0.2, (len(env_ids), self.num_dof), device=self.device)
        velocities = torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dof), device=self.device)

        self.dof_pos[env_ids] = tensor_clamp(self.initial_dof_pos[env_ids] + positions, self.dof_limits_lower,
                                             self.dof_limits_upper)
        self.dof_vel[env_ids] = velocities

        self.dof_pos_op[env_ids] = tensor_clamp(self.initial_dof_pos[env_ids] + positions, self.dof_limits_lower,
                                                self.dof_limits_upper)
        self.dof_vel_op[env_ids] = velocities

        env_ids_int32 = (torch.cat((self.actor_indices[env_ids], self.actor_indices_op[env_ids]))).to(dtype=torch.int32)
        agent_env_ids = expand_env_ids(env_ids, 2)

        rand_angle = torch.rand((len(env_ids),), device=self.device) * torch.pi * 2

        rand_pos = torch.ones((len(agent_env_ids), 2), device=self.device) * (
                self.borderline_space * torch.ones((len(agent_env_ids), 2), device=self.device) - torch.rand(
            (len(agent_env_ids), 2), device=self.device) * 2)
        rand_pos[0::2, 0] *= torch.cos(rand_angle)
        rand_pos[0::2, 1] *= torch.sin(rand_angle)
        rand_pos[1::2, 0] *= torch.cos(rand_angle + torch.pi)
        rand_pos[1::2, 1] *= torch.sin(rand_angle + torch.pi)
        rand_floats = torch_rand_float(-1.0, 1.0, (len(agent_env_ids), 3), device=self.device)
        rand_rotation = quat_from_angle_axis(rand_floats[:, 1] * np.pi, self.z_unit_tensor[agent_env_ids])
        rand_rotation2 = quat_from_angle_axis(rand_floats[:, 2] * np.pi, self.z_unit_tensor[agent_env_ids])
        self.root_states[agent_env_ids] = self.initial_root_states[agent_env_ids]
        self.root_states[agent_env_ids, :2] = rand_pos
        self.root_states[agent_env_ids[1::2], 3:7] = rand_rotation[1::2]
        self.root_states[agent_env_ids[0::2], 3:7] = rand_rotation2[0::2]
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.pos_before = self.root_states[0::2, :2].clone()

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def pre_physics_step(self, actions):
        # actions.shape = [num_envs * num_agents, num_actions], stacked as followed:
        # {[(agent1_act_1, agent1_act2)|(agent2_act1, agent2_act2)|...]_(env0),
        #  [(agent1_act_1, agent1_act2)|(agent2_act1, agent2_act2)|...]_(env1),
        #  ... }

        self.actions = actions.clone().to(self.device)
        self.actions = torch.cat((self.actions[:self.num_envs], self.actions[self.num_envs:]), dim=-1)

        # reshape [num_envs * num_agents, num_actions] to [num_envs, num_agents * num_actions]
        targets = self.actions

        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(targets))

    def post_physics_step(self):
        self.progress_buf += 1
        self.randomize_buf += 1

        resets = self.reset_buf.reshape(self.num_envs, 1).sum(dim=1)
        # print(resets)
        env_ids = (resets == 1).nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)
        self.pos_before = self.obs_buf[:self.num_envs, :2].clone()

    def get_number_of_agents(self):
        # train one agent with index 0
        return 1

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
        self.extras['win'][:] = 0
        self.extras['draw'][:] = 0


#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def expand_env_ids(env_ids, n_agents):
    # type: (Tensor, int) -> Tensor
    device = env_ids.device
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
    is_out = torch.sum(torch.square(obs_buf[:, 0:2]), dim=-1) >= borderline_space ** 2
    is_out_op = torch.sum(torch.square(obs_buf_op[:, 0:2]), dim=-1) >= borderline_space ** 2
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
    not_move_penalty = -10 * torch.exp(-torch.sum(torch.abs(torques), dim=1))
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
    obs = torch.cat(
        (root_states[:, :13], dof_pos_scaled, dof_vel * dof_vel_scale, root_states_op[:, :7],
         root_states[:, :2] - root_states_op[:, :2], torch.unsqueeze(root_states[:, 2] < termination_height, -1),
         torch.unsqueeze(root_states_op[:, 2] < termination_height, -1)), dim=-1)

    return obs


@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(quat_from_angle_axis(rand0 * np.pi, x_unit_tensor),
                    quat_from_angle_axis(rand1 * np.pi, y_unit_tensor))
