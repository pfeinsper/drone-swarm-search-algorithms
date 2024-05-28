import pathlib
from DSSE import CoverageDroneSwarmSearch
from DSSE.environment.wrappers import RetainDronePosWrapper, AllPositionsWrapper
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.tune.registry import register_env
from torch import nn
import torch
import numpy as np


class CNNModel(TorchModelV2, nn.Module):
    def __init__(
        self,
        obs_space,
        act_space,
        num_outputs,
        model_config,
        name,
        **kw,
    ):
        print("OBSSPACE: ", obs_space)
        TorchModelV2.__init__(
            self, obs_space, act_space, num_outputs, model_config, name, **kw
        )
        nn.Module.__init__(self)

        first_cnn_out = (obs_space[1].shape[0] - 3) + 1
        second_cnn_out = (first_cnn_out - 2) + 1
        flatten_size = 32 * second_cnn_out * second_cnn_out
        print("Cnn Dense layer input size: ", flatten_size)
        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=(3, 3),
            ),
            nn.Tanh(),
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=(2, 2),
            ),
            nn.Tanh(),
            nn.Flatten(),
            nn.Linear(flatten_size, 256),
            nn.Tanh(),
        )

        self.linear = nn.Sequential(
            nn.Linear(obs_space[0].shape[0], 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
        )

        self.join = nn.Sequential(
            nn.Linear(256 * 2, 256),
            nn.Tanh(),
        )
        
        self.policy_fn = nn.Linear(256, num_outputs)
        self.value_fn = nn.Linear(256, 1)

    def forward(self, input_dict, state, seq_lens):
        input_positions = input_dict["obs"][0].float()
        input_matrix = input_dict["obs"][1].float()

        input_matrix = input_matrix.unsqueeze(1)
        cnn_out = self.cnn(input_matrix)
        linear_out = self.linear(input_positions)

        value_input = torch.cat((cnn_out, linear_out), dim=1)
        value_input = self.join(value_input)
        
        self._value_out = self.value_fn(value_input)
        return self.policy_fn(value_input), state

    def value_function(self):
        return self._value_out.flatten()

def env_creator(args):
    print("-------------------------- ENV CREATOR --------------------------")
    N_AGENTS = 2
    # 6 hours of simulation, 600 radius
    env = CoverageDroneSwarmSearch(
        timestep_limit=200, drone_amount=N_AGENTS, prob_matrix_path="min_matrix.npy"
    )
    env = AllPositionsWrapper(env)
    grid_size = env.grid_size
    # positions = position_on_diagonal(grid_size, N_AGENTS)
    # positions = position_on_circle(grid_size, N_AGENTS, 2)
    positions = [
        (grid_size - 1, grid_size // 2),
        (0, grid_size // 2),
    ]
    env = RetainDronePosWrapper(env, positions)
    return env

def position_on_diagonal(grid_size, drone_amount):
    positions = []
    center = grid_size // 2
    for i in range(-drone_amount // 2, drone_amount // 2):
        positions.append((center + i, center + i))
    return positions

def position_on_circle(grid_size, drone_amount, radius):
    positions = []
    center = grid_size // 2
    angle_increment = 2 * np.pi / drone_amount

    for i in range(drone_amount):
        angle = i * angle_increment
        x = center + int(radius * np.cos(angle))
        y = center + int(radius * np.sin(angle))
        positions.append((x, y))

    return positions


if __name__ == "__main__":
    ray.init()

    env_name = "DSSE_Coverage"

    register_env(env_name, lambda config: ParallelPettingZooEnv(env_creator(config)))
    ModelCatalog.register_custom_model("CNNModel", CNNModel)

    config = (
        PPOConfig()
        .environment(env=env_name)
        .rollouts(num_rollout_workers=6, rollout_fragment_length="auto")
        .training(
            train_batch_size=8192 * 5,
            lr=6e-6,
            gamma=0.9999999,
            lambda_=0.9,
            use_gae=True,
            entropy_coeff=0.01,
            vf_clip_param=100000,
            sgd_minibatch_size=300,
            num_sgd_iter=10,
            model={
                "custom_model": "CNNModel",
                "_disable_preprocessor_api": True,
            },
        )
        .experimental(_disable_preprocessor_api=True)
        .debugging(log_level="ERROR")
        .framework(framework="torch")
        .resources(num_gpus=1)
    )

    curr_path = pathlib.Path().resolve()
    tune.run(
        "PPO",
        name="PPO_" + input("Exp name: "),
        # resume=True,
        stop={"timesteps_total": 20_000_000},
        checkpoint_freq=20,
        storage_path=f"{curr_path}/ray_res/" + env_name,
        config=config.to_dict(),
    )
