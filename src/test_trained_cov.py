import os
import pathlib
from recorder import PygameRecord
from DSSE import CoverageDroneSwarmSearch
from DSSE.environment.wrappers import RetainDronePosWrapper, AllPositionsWrapper
import ray
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch.nn as nn
import argparse
import torch



argparser = argparse.ArgumentParser()
argparser.add_argument("--checkpoint", type=str, required=True)
argparser.add_argument("--see", action="store_true", default=False)
args = argparser.parse_args()


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

        first_cnn_out = (obs_space[1].shape[0] - 8) + 1
        second_cnn_out = (first_cnn_out - 4) + 1
        flatten_size = 32 * second_cnn_out * second_cnn_out
        print("Cnn Dense layer input size: ", flatten_size)
        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=(8, 8),
            ),
            nn.Tanh(),
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=(4, 4),
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

# Register the model
ModelCatalog.register_custom_model("CNNModel", CNNModel)


def env_creator(args):
    print("-------------------------- ENV CREATOR --------------------------")
    N_AGENTS = 8
    # 6 hours of simulation, 600 radius
    env = CoverageDroneSwarmSearch(
        timestep_limit=180, drone_amount=N_AGENTS, prob_matrix_path="presim_20.npy", render_mode="human"
    )
    env = AllPositionsWrapper(env)
    grid_size = env.grid_size
    positions = position_on_diagonal(grid_size, N_AGENTS)
    env = RetainDronePosWrapper(env, positions)
    return env

def position_on_diagonal(grid_size, drone_amount):
    positions = []
    center = grid_size // 2
    for i in range(-drone_amount // 2, drone_amount // 2):
        positions.append((center + i, center + i))
    return positions

env = env_creator(None)
register_env("DSSE_Coverage", lambda config: ParallelPettingZooEnv(env_creator(config)))
ray.init()


checkpoint_path = args.checkpoint
PPOagent = PPO.from_checkpoint(checkpoint_path)

reward_sum = 0
i = 0

if args.see:
    obs, info = env.reset()
    with PygameRecord("test_trained.gif", 5) as rec:
        while env.agents:
            actions = {}
            for k, v in obs.items():
                actions[k] = PPOagent.compute_single_action(v, explore=False)
                # print(v)
            # action = PPOagent.compute_actions(obs)
            obs, rw, term, trunc, info = env.step(actions)
            reward_sum += sum(rw.values())
            i += 1
            rec.add_frame()
else:
    rewards = []
    founds = 0
    N_EVALS = 1000
    for _ in range(N_EVALS):
        obs, info = env.reset()
        reward_sum = 0
        while env.agents:
            actions = {}
            for k, v in obs.items():
                actions[k] = PPOagent.compute_single_action(v, explore=False)
                # print(v)
            # action = PPOagent.compute_actions(obs, explore=False)
            obs, rw, term, trunc, info = env.step(actions)
            reward_sum += sum(rw.values())
            i += 1
        rewards.append(reward_sum)
        for _, v in info.items():
            if v["Found"]:
                founds += 1
                break
    print("Average reward: ", sum(rewards) / N_EVALS)
    print("Found %: ", founds / N_EVALS)

print("Total reward: ", reward_sum)
print("Total steps: ", i)
print("Found: ", info)
env.close()
