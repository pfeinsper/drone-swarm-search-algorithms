import os
from DSSE import DroneSwarmSearch
from wrappers import AllPositionsWrapper, RetainDronePosWrapper
# from DSSE.environment.wrappers import AllPositionsWrapper, RetainDronePosWrapper
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
        print(
            "______________________________ MLPModel ________________________________"
        )
        print("OBSSPACE: ", obs_space)
        print("ACTSPACE: ", act_space)
        print("NUMOUTPUTS: ", num_outputs)
        TorchModelV2.__init__(
            self, obs_space, act_space, num_outputs, model_config, name, **kw
        )
        nn.Module.__init__(self)
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
                in_channels=16,
                out_channels=32,
                kernel_size=(4, 4),
                stride=(1, 1)
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=(3, 3),
                stride=(1, 1)
            ),
            # nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
        )
        grid_size = obs_space.shape[0]
        # Fully connected layers
        # F: (((W - K + 2P)/S) + 1)
        first_conv_output_size = (grid_size - 8) + 1
        second_conv_output_size = (first_conv_output_size - 4) + 1
        third_conv_output_size = (second_conv_output_size - 3) + 1
        self.fc1_input_size = third_conv_output_size * third_conv_output_size * 64
        # Apply a DENSE layer to the flattened CNN2 output
        self.fc1 = nn.Linear(self.fc1_input_size, 512)
        self.policy_fn = nn.Linear(512, num_outputs)
        self.value_fn = nn.Linear(512, 1)

    def forward(self, input_dict, state, seq_lens):
        input_ = input_dict["obs"]
        # Convert dims
        input_ = input_.unsqueeze(1)
        model_out = self.conv1(input_)
        model_out = self.fc1(model_out)
        self._value_out = self.value_fn(model_out)
        return self.policy_fn(model_out), state

    def value_function(self):
        return self._value_out.flatten()


def env_creator(args):
    env = DroneSwarmSearch(
        drone_amount=4,
        grid_size=20,
        dispersion_inc=0.1,
        person_initial_position=(10, 10),
    )
    env = MatrixEncodeWrapper(env)
    return env


if __name__ == "__main__":
    ray.init()

    env_name = "DSSE"

    register_env(env_name, lambda config: ParallelPettingZooEnv(env_creator(config)))
    ModelCatalog.register_custom_model("MLPModel", MLPModel)

    config = (
        PPOConfig()
        .environment(env=env_name)
        .rollouts(num_rollout_workers=5, rollout_fragment_length='auto')
        .training(
            train_batch_size=512,
            lr=2e-5,
            gamma=0.99999,
            lambda_=0.9,
            use_gae=True,
            clip_param=0.3,
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
        .debugging(log_level="ERROR")
        .framework(framework="torch")
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "1")))
    )
    config["_disable_preprocessor_api"] = False

    tune.run(
        "PPO",
        name="PPO",
        stop={"timesteps_total": 5000000 if not os.environ.get("CI") else 50000},
        checkpoint_freq=10,
        # storage_path="ray_res/" + env_name,
        config=config.to_dict(),
    )
