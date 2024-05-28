import os
import pathlib
from recorder import PygameRecord
from DSSE import CoverageDroneSwarmSearch
from DSSE.environment.wrappers import RetainDronePosWrapper, AllPositionsWrapper, AllFlattenWrapper
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


def env_creator(_):
    print("-------------------------- ENV CREATOR --------------------------")
    N_AGENTS = 2
    render_mode = "human" if args.see else "ansi"
    # 6 hours of simulation, 600 radius
    env = CoverageDroneSwarmSearch(
        timestep_limit=200, drone_amount=N_AGENTS, prob_matrix_path="min_matrix.npy", render_mode=render_mode
    )
    env = AllFlattenWrapper(env)
    grid_size = env.grid_size
    print("Grid size: ", grid_size)
    positions = [
        (0, grid_size // 2),
        (grid_size - 1, grid_size // 2),
    ]
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

def print_mean(values, name):
    print(f"Mean of {name}: ", sum(values) / len(values))

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
    print(info)
else:
    rewards = []
    cov_rate = []
    steps_needed = []
    repeated_cov = []

    N_EVALS = 500
    for _ in range(N_EVALS):
        i = 0
        obs, info = env.reset()
        reward_sum = 0
        while env.agents:
            actions = {}
            for k, v in obs.items():
                actions[k] = PPOagent.compute_single_action(v, explore=False)
            obs, rw, term, trunc, info = env.step(actions)
            reward_sum += sum(rw.values())
            i += 1
        rewards.append(reward_sum)
        steps_needed.append(i)
        cov_rate.append(info["drone0"]["coverage_rate"])
        repeated_cov.append(info["drone0"]["repeated_coverage"])
       
    print_mean(rewards, "rewards")
    print_mean(steps_needed, "steps needed")
    print_mean(cov_rate, "coverage rate")
    print_mean(repeated_cov, "repeated coverage")

print("Total reward: ", reward_sum)
print("Total steps: ", i)
print("Found: ", info)
env.close()
