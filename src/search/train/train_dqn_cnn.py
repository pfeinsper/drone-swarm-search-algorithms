import pathlib
from DSSE import DroneSwarmSearch
from DSSE.environment.wrappers import AllPositionsWrapper, RetainDronePosWrapper

# from DSSE.environment.wrappers import AllPositionsWrapper
import ray
from ray import tune
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.tune.registry import register_env
import torch
from torch import nn


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
    env = RetainDronePosWrapper(env, positions)
    return env


if __name__ == "__main__":
    ray.init()

    env_name = "DSSE"

    register_env(env_name, lambda config: ParallelPettingZooEnv(env_creator(config)))
    ModelCatalog.register_custom_model("CNNModel", CNNModel)

    natural_value = 512 / (14 * 20 * 1)
    config = (
        DQNConfig()
        .exploration(
            exploration_config={
                "type": "EpsilonGreedy",
                "initial_epsilon": 1.0,
                "final_epsilon": 0.15,
                "epsilon_timesteps": 400_000,
            }
        )
        .environment(env=env_name)
        .rollouts(num_rollout_workers=14, rollout_fragment_length=20)
        .framework("torch")
        .debugging(log_level="ERROR")
        .resources(num_gpus=1)
        .experimental(_disable_preprocessor_api=True)
        .training(
            lr=1e-4,
            gamma=0.9999999,
            tau=0.01,
            train_batch_size=512,
            model={
                "custom_model": "CNNModel",
                "_disable_preprocessor_api": True,
            },
            target_network_update_freq=500,
            double_q=False,
            training_intensity=natural_value,
            v_min=0,
            v_max=2,
        )
    )

    curr_path = pathlib.Path().resolve()
    tune.run(
        "DQN",
        name="DQN_DSSE",
        stop={"timesteps_total": 100_000_000},
        checkpoint_freq=200,
        storage_path=f"{curr_path}/ray_res/" + env_name,
        config=config.to_dict(),
    )

# Finalize Ray to free up resources
ray.shutdown()
