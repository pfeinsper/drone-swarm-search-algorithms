import os
import pathlib
import PygameRecord
from DSSE import DroneSwarmSearch
from wrappers import AllPositionsWrapper, RetainDronePosWrapper, TopNProbsWrapper
import ray
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from ray.rllib.algorithms.dqn import DQN
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch.nn as nn
import argparse
import random



argparser = argparse.ArgumentParser()
argparser.add_argument("--checkpoint", type=str, required=True)
argparser.add_argument("--see", action="store_true", default=False)
args = argparser.parse_args()

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

        self.model = nn.Sequential(
            nn.Linear(obs_space.shape[0], 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.policy_fn = nn.Linear(256, num_outputs)
        self.value_fn = nn.Linear(256, 1)

    def forward(self, input_dict, state, seq_lens):
        input_ = input_dict["obs"].float()
        value_input = self.model(input_)
        
        self._value_out = self.value_fn(value_input)
        return self.policy_fn(value_input), state

    def value_function(self):
        return self._value_out.flatten()


ModelCatalog.register_custom_model("MLPModel", MLPModel)

def ramdom_position(centro_x, centro_y, alcance, num_posicoes=4):
    # Gerar todas as posições possíveis dentro do alcance
    posicoes_possiveis = [(x, y) for x in range(centro_x - alcance, centro_x + alcance + 1)
                          for y in range(centro_y - alcance, centro_y + alcance + 1)]

    # Selecionar num_posicoes posições aleatoriamente das possíveis
    posicoes_aleatorias = random.sample(posicoes_possiveis, num_posicoes)

    return posicoes_aleatorias

# DEFINE HERE THE EXACT ENVIRONMENT YOU USED TO TRAIN THE AGENT
def env_creator(_):
    env = DroneSwarmSearch(
        drone_amount=4,
        grid_size=20,
        dispersion_inc=0.08,
        person_initial_position=(10, 10),
        person_amount=5,
        render_mode="ansi",
    )
    
    env = TopNProbsWrapper(env, 10)
    env = RetainDronePosWrapper(env, ramdom_position(10, 10, 3, 4))
    return env

env = env_creator(None)
register_env("DSSE", lambda config: ParallelPettingZooEnv(env_creator(config)))
ray.init()


checkpoint_path = args.checkpoint
DQNagent = DQN.from_checkpoint(checkpoint_path)

reward_sum = 0
i = 0

if args.see:
    obs, info = env.reset()
    with PygameRecord("test_trained.gif", 5) as rec:
        while env.agents:
            action = DQNagent.compute_actions(obs)
            obs, rw, term, trunc, info = env.step(action)
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
            action = DQNagent.compute_actions(obs, explore=False)
            obs, rw, term, trunc, info = env.step(action)
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
