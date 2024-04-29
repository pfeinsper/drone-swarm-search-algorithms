import os
import pathlib
from DSSE import DroneSwarmSearch
from wrappers import AllPositionsWrapper, RetainDronePosWrapper
# from DSSE.environment.wrappers import AllPositionsWrapper, RetainDronePosWrapper
import ray
from ray import air
from ray import tune
from ray.rllib.algorithms.dqn.dqn import DQNConfig
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
    env = AllPositionsWrapper(env)
    env = RetainDronePosWrapper(env, [(0, 0), (0, 19), (19, 0), (19, 19)])
    return env


if __name__ == "__main__":
    if not ray.is_initialized():
        ray.init()

    env_name = "DSSE"

    register_env(env_name, lambda config: ParallelPettingZooEnv(env_creator(config)))
    ModelCatalog.register_custom_model("MLPModel", MLPModel)

    config = (
        DQNConfig()
        .environment(env=env_name)
        .rollouts(num_rollout_workers=6, rollout_fragment_length='auto', num_envs_per_worker=2)
        .training(
            train_batch_size=512,
            grad_clip=None,
            target_network_update_freq=1,
            tau=0.005,
            gamma=0.99999,
            n_step=1,
            double_q=True,
            dueling=False,
            model={"custom_model": "MLPModel", "_disable_preprocessor_api": True},
            v_min=-800,
            v_max=800,
        )
        .exploration(
            exploration_config={
                "type": "EpsilonGreedy",
                "initial_epsilon": 1.0,
                "final_epsilon": 0.05,
                "epsilon_timesteps": 350_000,
            }
        )
        .debugging(log_level="ERROR")
        .framework(framework="torch")
        .resources(num_gpus=0)
        .experimental(_disable_preprocessor_api=False)
    )
    config["_disable_preprocessor_api"] = False

    # curr_path = pathlib.Path().resolve()
    # run_config = air.RunConfig(
    #     stop={"timesteps_total": 10_000_000 if not os.environ.get("CI") else 50000},
    #     storage_path=f"{curr_path}/ray_res/" + env_name,
    #     checkpoint_config=air.CheckpointConfig(checkpoint_frequency=10),
    # )
    # tune.Tuner(
    #     "DQN",
    #     run_config=run_config,
    #     param_space=config.to_dict()
    # ).fit()
    
    tune.run(
        "DQN",
        name="DQN",
        stop={"timesteps_total": 5000000 if not os.environ.get("CI") else 50000},
        checkpoint_freq=10,
        # storage_path="ray_res/" + env_name,
        config=config.to_dict(),
    )
    
    
