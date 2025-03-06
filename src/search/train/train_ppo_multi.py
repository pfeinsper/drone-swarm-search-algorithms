import os
import pathlib
from DSSE import DroneSwarmSearch
from DSSE.environment.wrappers import AllPositionsWrapper
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.tune.registry import register_env
import torch
from torch import nn


class MLPModel(TorchModelV2, nn.Module):
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

        # F: (((W - K + 2P)/S) + 1)
        grid_size = obs_space[1].shape[0]
        first_conv_output_size = (grid_size - 8) + 1
        second_conv_output_size = (first_conv_output_size - 4) + 1
        third_conv_output_size = (second_conv_output_size - 3) + 1
        self.fc1_input_size = third_conv_output_size * third_conv_output_size * 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=(8, 8),
                stride=(1, 1),
            ),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(
                in_channels=16, out_channels=32, kernel_size=(4, 4), stride=(1, 1)
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1)
            ),
            # nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(self.fc1_input_size, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
        )

        self.fc_scalar = nn.Sequential(
            nn.Linear(obs_space[0].shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
        )

        self.unifier = nn.Sequential(
            nn.Linear(512 * 2, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
        )
        self.policy_fn = nn.Linear(512, num_outputs)
        self.value_fn = nn.Linear(512, 1)

    def forward(self, input_dict, state, seq_lens):
        input_ = input_dict["obs"]
        # Convert dims
        input_cnn = input_[1].unsqueeze(1)
        model_out = self.conv1(input_cnn)

        scalar_input = input_[0].float()
        scalar_out = self.fc_scalar(scalar_input)

        value_input = torch.cat((model_out, scalar_out), -1)
        value_input = self.unifier(value_input)

        self._value_out = self.value_fn(value_input)
        return self.policy_fn(value_input), state

    def value_function(self):
        return self._value_out.flatten()


def env_creator(args):
    env = DroneSwarmSearch(
        drone_amount=4,
        grid_size=20,
        dispersion_inc=0.1,
        person_initial_position=(10, 10),
    )
    env = AllPositionsWrapper(env)
    return env


if __name__ == "__main__":
    ray.init()

    env_name = "DSSE"

    register_env(env_name, lambda config: ParallelPettingZooEnv(env_creator(config)))
    ModelCatalog.register_custom_model("MLPModel", MLPModel)

    config = (
        PPOConfig()
        .environment(env=env_name)
        .rollouts(num_rollout_workers=4, rollout_fragment_length=128)
        .training(
            train_batch_size=512,
            lr=4e-5,
            gamma=0.99999,
            lambda_=0.9,
            use_gae=True,
            clip_param=0.4,
            grad_clip=None,
            entropy_coeff=0.1,
            vf_loss_coeff=0.25,
            vf_clip_param=420,
            sgd_minibatch_size=64,
            num_sgd_iter=10,
            model={
                "custom_model": "MLPModel",
                "_disable_preprocessor_api": True,
            },
        )
        .experimental(_disable_preprocessor_api=True)
        .debugging(log_level="ERROR")
        .framework(framework="torch")
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "1")))
    )

    curr_path = pathlib.Path().resolve()
    tune.run(
        "PPO",
        name="PPO",
        stop={"timesteps_total": 5000000 if not os.environ.get("CI") else 50000},
        checkpoint_freq=10,
        # local_dir="ray_results/" + env_name,
        storage_path=f"{curr_path}/ray_res/" + env_name,
        config=config.to_dict(),
    )
