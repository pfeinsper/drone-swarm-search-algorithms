import os
import pathlib
from DSSE import DroneSwarmSearch
from DSSE.environment.wrappers import RetainDronePosWrapper, AllPositionsWrapper
from DSSE.environment.wrappers.communication_wrapper import CommunicationWrapper
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.tune.registry import register_env
from torch import nn
import torch


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

        flatten_size = (
            32 * (obs_space[1].shape[0] - 7 - 3) * (obs_space[1].shape[1] - 7 - 3)
        )
        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=(8, 8),
                stride=(1, 1),
            ),
            nn.Tanh(),
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=(4, 4),
                stride=(1, 1),
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
    env = DroneSwarmSearch(
        drone_amount=4,
        grid_size=40,
        dispersion_inc=0.1,
        person_initial_position=(20, 20),
    )
    positions = [
        (20, 0),
        (20, 39),
        (0, 20),
        (39, 20),
    ]
    env = AllPositionsWrapper(env)
    env = CommunicationWrapper(env, n_steps=12)
    env = RetainDronePosWrapper(env, positions)
    return env


if __name__ == "__main__":
    ray.init()

    env_name = "DSSE"

    register_env(env_name, lambda config: ParallelPettingZooEnv(env_creator(config)))
    ModelCatalog.register_custom_model("CNNModel", CNNModel)

    config = (
        PPOConfig()
        .environment(env=env_name)
        .rollouts(
            num_rollout_workers=6, rollout_fragment_length="auto", num_envs_per_worker=2
        )
        .training(
            train_batch_size=8192,
            lr=1e-5,
            gamma=0.9999999,
            lambda_=0.9,
            use_gae=True,
            # clip_param=0.3,
            # grad_clip=None,
            entropy_coeff=0.01,
            # vf_loss_coeff=0.25,
            # vf_clip_param=10,
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
        name="PPO_COMM_WRAPPER",
        stop={
            "timesteps_total": 20_000_000 if not os.environ.get("CI") else 50000,
            "episode_reward_mean": 1.75,
        },
        checkpoint_freq=10,
        storage_path=f"{curr_path}/ray_res/" + env_name,
        config=config.to_dict(),
    )
