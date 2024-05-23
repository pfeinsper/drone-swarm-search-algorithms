from recorder import PygameRecord
from DSSE import DroneSwarmSearch
from DSSE.environment.wrappers import RetainDronePosWrapper, AllPositionsWrapper
from DSSE.environment.wrappers.communication_wrapper import CommunicationWrapper
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch.nn as nn
import argparse
import torch
import numpy as np



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

        flatten_size = 32 * (obs_space[1].shape[0] - 7 - 3) * (obs_space[1].shape[1] - 7 - 3)
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



# ModelCatalog.register_custom_model("MLPModel", MLPModel)
ModelCatalog.register_custom_model("CNNModel", CNNModel)

# DEFINE HERE THE EXACT ENVIRONMENT YOU USED TO TRAIN THE AGENT
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

env = env_creator(None)
register_env("DSSE", lambda config: ParallelPettingZooEnv(env_creator(config)))
ray.init()


checkpoint_path = args.checkpoint
PPOagent = PPO.from_checkpoint(checkpoint_path)


if args.see:
    i = 0
    reward_sum = 0
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
    print(info)
    print(reward_sum)
else:
    rewards = []
    actions_stat = []
    founds = 0
    N_EVALS = 5_000
    for epoch in range(N_EVALS):
        print(epoch)
        obs, info = env.reset()
        i = 0
        reward_sum = 0
        while env.agents:
            actions = {}
            for k, v in obs.items():
                actions[k] = PPOagent.compute_single_action(v, explore=False)
            obs, rw, term, trunc, info = env.step(actions)
            reward_sum += sum(rw.values())
            i += 1
        rewards.append(reward_sum)
        actions_stat.append(i)

        for _, v in info.items():
            if v["Found"]:
                founds += 1
                break
    print("Average reward: ", sum(rewards) / N_EVALS)
    print("Average Actions: ", sum(actions_stat) / N_EVALS)
    print("Median of actions: ", np.median(actions_stat))
    print("Found %: ", founds / N_EVALS)

env.close()
