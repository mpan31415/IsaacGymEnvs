import isaacgym

import torch
from rl_games.algos_torch.models import ModelA2CContinuousLogStd
from rl_games.torch_runner import Runner
from rl_games.algos_torch import model_builder
from omegaconf import OmegaConf

import yaml
import gym
import isaacgymenvs
from rl_games.common import env_configurations, vecenv
from isaacgymenvs.tasks import isaacgym_task_map

# Load config used to build the network
# train_config_path1 = "/home/gymuser/RL/IsaacGymEnvs/isaacgymenvs/cfg/eval/AllegroFrankaPPO.yaml"
# train_config_path2 = "/home/gymuser/RL/IsaacGymEnvs/isaacgymenvs/cfg/eval/AllegroFrankaLSTMPPO.yaml"
# train_config_path3 = "/home/gymuser/RL/IsaacGymEnvs/isaacgymenvs/cfg/eval/AllegroFrankaTwoArmsLSTMPPO.yaml"

# train_cfg1 = OmegaConf.load(train_config_path1)
# train_cfg2 = OmegaConf.load(train_config_path2)
# train_cfg3 = OmegaConf.load(train_config_path3)
# train_cfg = OmegaConf.merge(train_cfg1, train_cfg2, train_cfg3)


env_config_path = "/home/gymuser/RL/IsaacGymEnvs/isaacgymenvs/cfg/config.yaml"
cfg = OmegaConf.load(env_config_path)

def create_isaacgym_env(**kwargs):
    envs = isaacgymenvs.make(
        cfg.seed, 
        cfg.task_name, 
        cfg.task.env.numEnvs, 
        cfg.sim_device,
        cfg.rl_device,
        cfg.graphics_device_id,
        cfg.headless,
        cfg.multi_gpu,
        cfg.capture_video,
        cfg.force_render,
        cfg,
        **kwargs,
    )
    return envs

env_configurations.register('rlgpu', {
    'vecenv_type': 'RLGPU',
    'env_creator': lambda **kwargs: create_isaacgym_env(**kwargs),
})

ige_env_cls = isaacgym_task_map["AllegroFrankaTwoArms"]
dict_cls = ige_env_cls.dict_obs_cls if hasattr(ige_env_cls, 'dict_obs_cls') and ige_env_cls.dict_obs_cls else False

if dict_cls:
    
    obs_spec = {}
    actor_net_cfg = cfg.train.params.network
    obs_spec['obs'] = {'names': list(actor_net_cfg.inputs.keys()), 'concat': not actor_net_cfg.name == "complex_net", 'space_name': 'observation_space'}
    if "central_value_config" in cfg.train.params.config:
        critic_net_cfg = cfg.train.params.config.central_value_config.network
        obs_spec['states'] = {'names': list(critic_net_cfg.inputs.keys()), 'concat': not critic_net_cfg.name == "complex_net", 'space_name': 'state_space'}
    
    vecenv.register('RLGPU', lambda config_name, num_actors, **kwargs: ComplexObsRLGPUEnv(config_name, num_actors, obs_spec, **kwargs))
else:

    vecenv.register('RLGPU', lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))

# Load the checkpoint
checkpoint_path = "/home/gymuser/RL/AllegroFrankaTwoArmsLSTMPPO_18-11-37-57/nn/last_AllegroFrankaTwoArmsLSTMPPO_ep_3000_rew__197.5_.pth"
checkpoint = torch.load(checkpoint_path, map_location='cpu')
model_checkpoint_dict = checkpoint["model"]

print(f"model_checkpoint_dict keys: {model_checkpoint_dict.keys()}")

eval_config_path = "/home/gymuser/RL/IsaacGymEnvs/isaacgymenvs/cfg/eval/AllegroFrankaTwoArmsLSTMPPOEval.yaml"
with open(eval_config_path, 'r') as stream:
    config = yaml.safe_load(stream)
    runner = Runner()
    runner.load(config)
    agent = runner.create_player()
    # agent.restore(trained_network)



# # Create input dictionary for rl_games network builder
# train_params_dict = train_cfg['params']
# builder = model_builder.ModelBuilder()
# model : ModelA2CContinuousLogStd = builder.load(train_params_dict)




# # You might need to access 'model_state_dict' key if it's nested
# model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
# model.eval()


# print("=" * 100)
# print("Successful!")
# print("=" * 100)