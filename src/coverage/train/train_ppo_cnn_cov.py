import pathlib
from DSSE import CoverageDroneSwarmSearch
from DSSE.environment.wrappers import RetainDronePosWrapper, AllPositionsWrapper
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
import numpy as np
from src.models.ppo_cnn import PpoCnnModel
from src.models.cnn_config import CNNConfig


def env_creator(_):
    print("-------------------------- ENV CREATOR --------------------------")
    N_AGENTS = 2
    # 6 hours of simulation, 600 radius
    env = CoverageDroneSwarmSearch(
        timestep_limit=200, drone_amount=N_AGENTS, prob_matrix_path="min_matrix.npy"
    )
    env = AllPositionsWrapper(env)
    grid_size = env.grid_size
    # positions = position_on_diagonal(grid_size, N_AGENTS)
    # positions = position_on_circle(grid_size, N_AGENTS, 2)
    positions = [
        (grid_size - 1, grid_size // 2),
        (0, grid_size // 2),
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


def main(args):
    ray.init()

    env_name = "DSSE_Coverage"

    register_env(env_name, lambda config: ParallelPettingZooEnv(env_creator(config)))
    model = PpoCnnModel
    model.CONFIG = CNNConfig(kernel_sizes=[(3, 3), (2, 2)])
    ModelCatalog.register_custom_model(model.NAME, model)

    config = (
        PPOConfig()
        .environment(env=env_name)
        .rollouts(num_rollout_workers=6, rollout_fragment_length="auto")
        .training(
            train_batch_size=8192 * 5,
            lr=6e-6,
            gamma=0.9999999,
            lambda_=0.9,
            use_gae=True,
            entropy_coeff=0.01,
            vf_clip_param=100000,
            minibatch_size=300,
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
        name="PPO_" + input("Exp name: "),
        # resume=True,
        stop={"timesteps_total": 20_000_000},
        checkpoint_freq=20,
        storage_path=f"{curr_path}/ray_res/" + env_name,
        config=config.to_dict(),
    )
