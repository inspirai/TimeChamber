import os
import sys
import yaml

from isaacgym import gymapi
from isaacgym import gymutil

import numpy as np
import random
import torch

SIM_TIMESTEP = 1.0 / 60.0

def parse_sim_params(args, cfg):
    # initialize sim
    sim_params = gymapi.SimParams()
    sim_params.dt = SIM_TIMESTEP
    sim_params.num_client_threads = args.num_subscenes
    if args.physics_engine == "flex":
        if args.device_type != "cpu":
            print("WARNING: Using Flex with GPU instead of PHYSX!")
        sim_params.flex.shape_collision_margin = 0.01
        sim_params.flex.num_outer_iterations = 4
        sim_params.flex.num_inner_iterations = 10
    elif args.physics_engine == "physx":
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 0
        sim_params.physx.num_threads = 4
        sim_params.physx.use_gpu = args.use_gpu
        sim_params.physx.num_subscenes = args.num_subscenes
        sim_params.physx.max_gpu_contact_pairs = 8 * 1024 * 1024

    sim_params.use_gpu_pipeline = args.use_gpu_pipeline
    sim_params.physx.use_gpu = args.use_gpu

    # if sim options are provided in cfg, parse them and update/override above:
    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    # Override num_threads if passed on the command line
    if args.physics_engine == "physx" and args.num_threads > 0:
        sim_params.physx.num_threads = args.num_threads

    return sim_params