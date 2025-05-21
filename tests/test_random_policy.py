import isaacgym

from isaacgymenvs.tasks.allegro_franka.allegro_franka_two_arms_reorientation import AllegroFrankaTwoArmsReorientation
from isaacgymenvs.tasks.allegro_kuka.allegro_kuka_two_arms_reorientation import AllegroKukaTwoArmsReorientation
from rl_games.common.tr_helpers import unsqueeze_obs

import torch
import torch.nn as nn

from omegaconf import OmegaConf
from enum import Enum

from isaacgym import gymapi


class RobotType(Enum):
    FRANKA = 0
    KUKA = 1

class ActionType(Enum):
    ZERO = 0
    RANDOM = 1
    POLICY = 2


###################################
robot_type = RobotType.FRANKA
action_type = ActionType.ZERO
num_envs = 16
###################################


# ---- Load task config ----
if robot_type == RobotType.FRANKA:
    task_config_path1 = "/home/gymuser/RL/IsaacGymEnvs/isaacgymenvs/cfg/task/AllegroFranka.yaml"
    task_config_path2 = "/home/gymuser/RL/IsaacGymEnvs/isaacgymenvs/cfg/task/AllegroFrankaLSTM.yaml"
    task_config_path3 = "/home/gymuser/RL/IsaacGymEnvs/isaacgymenvs/cfg/task/AllegroFrankaTwoArmsLSTM.yaml"
elif robot_type == RobotType.KUKA:
    task_config_path1 = "/home/gymuser/RL/IsaacGymEnvs/isaacgymenvs/cfg/task/AllegroKuka.yaml"
    task_config_path2 = "/home/gymuser/RL/IsaacGymEnvs/isaacgymenvs/cfg/task/AllegroKukaLSTM.yaml"
    task_config_path3 = "/home/gymuser/RL/IsaacGymEnvs/isaacgymenvs/cfg/task/AllegroKukaTwoArmsLSTM.yaml"

task_cfg1 = OmegaConf.load(task_config_path1)
task_cfg2 = OmegaConf.load(task_config_path2)
task_cfg3 = OmegaConf.load(task_config_path3)
task_cfg = OmegaConf.merge(task_cfg1, task_cfg2, task_cfg3)
task_cfg['sim']['use_gpu_pipeline'] = False
task_cfg['env']['numEnvs'] = num_envs
task_cfg['physics_engine'] = 'physx'
task_cfg['sim']['physx']['num_subscenes'] = 4

# ---- Load train config ----
if robot_type == RobotType.FRANKA:
    train_config_path1 = "/home/gymuser/RL/IsaacGymEnvs/isaacgymenvs/cfg/train/AllegroFrankaPPO.yaml"
    train_config_path2 = "/home/gymuser/RL/IsaacGymEnvs/isaacgymenvs/cfg/train/AllegroFrankaLSTMPPO.yaml"
    train_config_path3 = "/home/gymuser/RL/IsaacGymEnvs/isaacgymenvs/cfg/train/AllegroFrankaTwoArmsLSTMPPO.yaml"
elif robot_type == RobotType.KUKA:
    train_config_path1 = "/home/gymuser/RL/IsaacGymEnvs/isaacgymenvs/cfg/train/AllegroKukaPPO.yaml"
    train_config_path2 = "/home/gymuser/RL/IsaacGymEnvs/isaacgymenvs/cfg/train/AllegroKukaLSTMPPO.yaml"
    train_config_path3 = "/home/gymuser/RL/IsaacGymEnvs/isaacgymenvs/cfg/train/AllegroKukaTwoArmsLSTMPPO.yaml"

train_cfg1 = OmegaConf.load(train_config_path1)
train_cfg2 = OmegaConf.load(train_config_path2)
train_cfg3 = OmegaConf.load(train_config_path3)
train_cfg = OmegaConf.merge(train_cfg1, train_cfg2, train_cfg3)

device = torch.device('cpu')  # MX130 can't train, but infer works

# ---- Create environment instance ----
if robot_type == RobotType.FRANKA:
    env = AllegroFrankaTwoArmsReorientation(task_cfg, rl_device="cpu", sim_device="cpu:0",
                                            graphics_device_id=0, headless=False,
                                            virtual_screen_capture=False, force_render=False)
elif robot_type == RobotType.KUKA:
    env = AllegroKukaTwoArmsReorientation(task_cfg, rl_device="cpu", sim_device="cpu:0",
                                            graphics_device_id=0, headless=False,
                                            virtual_screen_capture=False, force_render=False)

obs_dim = env.num_obs
act_dim = env.num_acts
print(f"obs_dim: {obs_dim}, act_dim: {act_dim}")
use_lstm = train_cfg['params']['network'].get('rnn', {}).get('name', '') == 'lstm'
print(f"use_lstm: {use_lstm}")

# ---- Load and build the policy network ----
checkpoints_dir = "/home/gymuser/RL/IsaacGymEnvs/isaacgymenvs/checkpoints/"
checkpoint_name = "AllegroFrankaTwoArmsLSTMPPO_18-11-37-57/nn/last_AllegroFrankaTwoArmsLSTMPPO_ep_3000_rew__197.5_.pth"
checkpoint = torch.load(checkpoints_dir + checkpoint_name, map_location='cpu')
print(checkpoint.keys())


class LSTMPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim, lstm_hidden_size=768, mlp_units=[768, 512, 256], activation='elu'):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.lstm_hidden_size = lstm_hidden_size
        self.activation_fn = nn.ELU() if activation == 'elu' else nn.ReLU()

        # LSTM comes before MLP
        self.lstm = nn.LSTM(input_size=obs_dim, hidden_size=lstm_hidden_size, num_layers=1, batch_first=True)
        self.layer_norm = nn.LayerNorm(lstm_hidden_size)

        # MLP after LSTM
        mlp_layers = []
        in_dim = lstm_hidden_size
        for out_dim in mlp_units:
            mlp_layers.append(nn.Linear(in_dim, out_dim))
            mlp_layers.append(self.activation_fn)
            in_dim = out_dim
        self.mlp = nn.Sequential(*mlp_layers)

        # Output action layer (assume continuous action space)
        self.action_head = nn.Linear(in_dim, action_dim)

    def forward(self, obs, hidden_state=None):
        # obs shape: [batch, obs_dim] -> we need [batch, seq_len, obs_dim]
        obs = obs.unsqueeze(1)  # add sequence dimension
        lstm_out, hidden_state = self.lstm(obs, hidden_state)
        lstm_out = lstm_out[:, -1, :]  # Take last output
        lstm_out = self.layer_norm(lstm_out)

        x = self.mlp(lstm_out)
        action = self.action_head(x)
        return action, hidden_state
    

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate model and hidden state
policy = LSTMPolicy(obs_dim=obs_dim, action_dim=act_dim).to(device)
hidden_state = (torch.zeros(1, 1, 768).to(device),  # h_0
                torch.zeros(1, 1, 768).to(device))  # c_0
policy.eval()  # Important: turn off dropout, etc. for inference

# ---- Rollout loop ----
num_sim_steps = 5000

for i in range(1, num_sim_steps+1):

    if i % 200 == 0:
        print(f"rollout {i}/{num_sim_steps}")

    # Get observations
    obs_buf, _ = env.compute_observations()
    obs = unsqueeze_obs(obs_buf[0]).float()  # shape [1, obs_dim]

    # Get action
    if action_type == ActionType.ZERO:
        action = env.zero_actions()
    elif action_type == ActionType.RANDOM:
        action = torch.rand_like(env.zero_actions()).to(device)
    elif action_type == ActionType.POLICY:
        with torch.no_grad():
            action, hidden_state = policy(obs, hidden_state)
    else:
        raise ValueError("Invalid action type")

    # apply action
    env.step(action)

    # Sync viewer
    if env.viewer:
        env.gym.step_graphics(env.sim)
        env.gym.draw_viewer(env.viewer, env.sim, True)
        env.gym.sync_frame_time(env.sim)

    if env.gym.query_viewer_has_closed(env.viewer):
        break

# Cleanup
env.gym.destroy_viewer(env.viewer)
env.gym.destroy_sim(env.sim)
