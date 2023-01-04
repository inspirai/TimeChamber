## Environments

We provide a detailed description of the environment here.

### Humanoid Strike

Humanoid Strike is a 3D environment with two simulated humanoid physics characters. Each character is equipped with a sword and shield with 37 degrees-of-freedom.
The game will be restarted if one agent goes outside the arena or the game reaches the maximum episode steps. We measure how much the player damaged the opponent and how much the player was damaged by the opponent in the terminated step to determine the winner.

#### <span id="obs1-1">Low-Level Observation Space</span>

|  Index  |          Description           |
|:-------:|:------------------------------:|
|  0   |           Height of the root from the ground.            |
|  1 - 48  |         Position of the body in the character’s local coordinate frame.         |
|  49 - 150  |      Rotation of the body in the character’s local coordinate frame.      |
| 151 - 201 |      Linear velocity of the root in the character’s local coordinate frame.       |
| 202 - 252 |      angular velocity of the root in the character’s local coordinate frame.          |


#### <span id="obs1-2">High-Level Observation Space</span>

|  Index  |          Description           |
|:-------:|:------------------------------:|
|  0 - 1  |    relative distance from the borderline            |
|  2 - 4  |    relative distance from the opponent          |
|  5 - 10  |      Rotation of the opponent's root in the character’s local coordinate frame.      |
| 11 - 13 |      Linear velocity of the opponent'root in the character’s local coordinate frame.       |
| 14 - 16 |      angular velocity of the opponent'root in the character’s local coordinate frame.         |
| 17 - 19 |      relative distance between ego agent and opponent's sword         |
| 20 - 22 |      Linear velocity of the opponent' sword in the character’s local coordinate frame.          |
| 23 - 25 |      relative distance between ego agent' shield and opponent's sword        |
| 26 - 28 | relative velocity between ego agent' shield and opponent's sword |
|   29 - 31    |   relative distance between ego agent' sword and opponent's torse    |
|   32 - 34    | relative velocity between ego agent' sword and opponent's torse  |
|   35 - 37    |   relative distance between ego agent' sword and opponent's head    |
|   38 - 40    | relative velocity between ego agent' sword and opponent's head  |
|   41 - 43    |   relative distance between ego agent' sword and opponent's right arm    |
|   44 - 46    | relative distance between ego agent' sword and opponent's right thigh  |
|   47 - 49    | relative distance between ego agent' sword and opponent's left thigh  |


#### <span id="action1-1">Low-Level Action Space</span>

| Index |    Description    |
|:-----:|:-----------------:|
| 0 - 30 | target rotations  of each character’s joints |

#### <span id="action1-2">High-Level Action Space</span>

| Index |    Description    |
|:-----:|:-----------------:|
| 0 - 63 | latent skill variables |

#### <span id="r1">Rewards</span>

The weights of reward components are as follows:

```python
op_fall_reward_w = 200.0
ego_fall_out_reward_w = 50.0
shield_to_sword_pos_reward_w = 1.0
damage_reward_w = 8.0
sword_to_op_reward_w = 0.8
reward_energy_w = 3.0
reward_strike_vel_acc_w = 3.0
reward_face_w = 4.0
reward_foot_to_op_w = 10.0
reward_kick_w = 2.0
```


### Ant Sumo

Ant Sumo is a 3D environment with simulated physics that allows pairs of ant agents to compete against each other.
To win, the agent has to push the opponent out of the ring. Every agent has 100 hp . Each step, If the agent's body
touches the ground, its hp will be reduced by 1.The agent whose hp becomes 0 will be eliminated.

#### <span id="obs2">Observation Space</span>

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

#### <span id="action2">Action Space</span>

| Index |    Description    |
|:-----:|:-----------------:|
| 0 - 7 | self dof position |

#### <span id="r2">Rewards</span>

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

#### <span id="obs3">Observation Space</span>

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

#### <span id="action3">Action Space</span>

| Index |    Description    |
|:-----:|:-----------------:|
| 0 - 7 | self dof position |

#### <span id="r3">Rewards</span>

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