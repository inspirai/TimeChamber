## Environments

We provide a detailed description of the environment here.

### Ant Sumo

Ant Sumo is a 3D environment with simulated physics that allows pairs of ant agents to compete against each other.
To win, the agent has to push the opponent out of the ring. Every agent has 100 hp . Each step, If the agent's body
touches the ground, its hp will be reduced by 1.The agent whose hp becomes 0 will be eliminated.

#### <span id="obs1">Observation Space</span>

|  Index  |          Description           |
|:-------:|:------------------------------:|
|  0 - 2  |           self pose            |
|  3 - 6  |         self rotation          |
|  7 - 9  |      self linear velocity      |
| 10 - 12 |      self angle velocity       |
| 13 - 20 |          self dof pos          |
| 21 - 28 |       self dof velocity        |
| 29 - 31 |         opponent pose          |
| 32 - 35 |       opponent rotation        |
| 36 - 37 | self-opponent pose vector(x,y) |
|   38    |   is self body touch ground    |
|   39    | is opponent body touch ground  |

#### <span id="action1">Action Space</span>

| Index |    Description    |
|:-----:|:-----------------:|
| 0 - 7 | self dof position |

#### <span id="r1">Rewards</span>

The reward consists of two parts:sparse reward and dense reward.

```python
win_reward = 2000
lose_penalty = -2000
draw_penalty = -1000
dense_reward_scale = 1.
dof_at_limit_cost = torch.sum(obs_buf[:, 13:21] > 0.99, dim=-1) * joints_at_limit_cost_scale
push_reward = -push_scale * torch.exp(-torch.linalg.norm(obs_buf_op[:, :2], dim=-1))
action_cost_penalty = torch.sum(torch.square(torques), dim=1) * action_cost_scale
not_move_penalty = -10 * torch.exp(-torch.sum(torch.abs(torques), dim=1))
dense_reward = move_reward + dof_at_limit_cost + push_reward + action_cost_penalty + not_move_penalty
total_reward = win_reward + lose_penalty + draw_penalty + dense_reward * dense_reward_scale
```

### Ant Battle

Ant Battle is an expanded environment of Ant Sumo. It supports more than two agents competing against with
each other. The battle ring radius will shrink, the agent going out of the ring will be eliminated.

#### <span id="obs2">Observation Space</span>

|  Index  |              Description               |
|:-------:|:--------------------------------------:|
|  0 - 2  |               self pose                |
|  3 - 6  |             self rotation              |
|  7 - 9  |          self linear velocity          |
| 10 - 12 |          self angle velocity           |
| 13 - 20 |              self dof pos              |
| 21 - 28 |           self dof velocity            |
|   29    |    border radius-self dis to centre    |
|   30    |             border radius              |
|   31    |       is self body touch ground        |
| 32 - 34 |            opponent_1 pose             |
| 35 - 38 |          opponent_1 rotation           |
| 39 - 40 |    self-opponent_1 pose vector(x,y)    |
| 41 - 48 |          opponent_1 dof pose           |
| 49 - 56 |        opponent_1 dof velocity         |
|   57    | border radius-opponent_1 dis to centre |
|   58    |    is opponent_1 body touch ground     |
|   ...   |                  ...                   |

#### <span id="action2">Action Space</span>

| Index |    Description    |
|:-----:|:-----------------:|
| 0 - 7 | self dof position |

#### <span id="r2">Rewards</span>

The reward consists of two parts:sparse reward and dense reward.

```python
win_reward_scale = 2000
reward_per_rank = 2 * win_reward_scale / (num_agents - 1)
sparse_reward = sparse_reward * (win_reward_scale - (nxt_rank[:, 0] - 1) * reward_per_rank)
stay_in_center_reward = stay_in_center_reward_scale * torch.exp(-torch.linalg.norm(obs[0, :, :2], dim=-1))
dof_at_limit_cost = torch.sum(obs[0, :, 13:21] > 0.99, dim=-1) * joints_at_limit_cost_scale
action_cost_penalty = torch.sum(torch.square(torques), dim=1) * action_cost_scale
not_move_penalty = torch.exp(-torch.sum(torch.abs(torques), dim=1))
dense_reward = dof_at_limit_cost + action_cost_penalty + not_move_penalty + stay_in_center_reward
total_reward = sparse_reward + dense_reward * dense_reward_scale
```