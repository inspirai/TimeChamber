import isaacgym
import isaacgymenvs
import torch

envs = isaacgymenvs.make(
    seed=0,
    task="MA_Ant_Battle",
    num_envs=2,
    sim_device="cuda:0",
    rl_device="cuda:0",
    graphics_device_id=0
)
print("Observation space is", envs.observation_space)
print("Action space is", envs.action_space)
obs = envs.reset()
for _ in range(20000):
    obs, reward, done, info = envs.step(
        torch.rand((2, 24,), device="cuda:0")
    )
