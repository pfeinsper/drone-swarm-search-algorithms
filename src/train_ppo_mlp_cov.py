import pathlib
from DSSE import CoverageDroneSwarmSearch
from DSSE.environment.wrappers import RetainDronePosWrapper, AllFlattenWrapper
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env
from torch import nn
import torch
import numpy as np


def env_creator(args):
    print("-------------------------- ENV CREATOR --------------------------")
    N_AGENTS = 2
    # 6 hours of simulation, 600 radius
    env = CoverageDroneSwarmSearch(
        timestep_limit=200, drone_amount=N_AGENTS, prob_matrix_path="min_matrix.npy"
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

    config = (
        PPOConfig()
        .environment(env=env_name)
        .rollouts(num_rollout_workers=6, rollout_fragment_length="auto", num_envs_per_worker=4)
        .training(
            train_batch_size=8192 * 3,
            lr=8e-6,
            gamma=0.9999999,
            lambda_=0.9,
            use_gae=True,
            entropy_coeff=0.01,
            vf_clip_param=100000,
            sgd_minibatch_size=300,
            num_sgd_iter=10,
            model={
                "fcnet_hiddens": [512, 256],
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
        stop={"timesteps_total": 40_000_000},
        checkpoint_freq=25,
        storage_path=f"{curr_path}/ray_res/" + env_name,
        config=config.to_dict(),
    )
