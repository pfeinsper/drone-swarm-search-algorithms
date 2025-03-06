from DSSE import CoverageDroneSwarmSearch
from DSSE.environment.wrappers import RetainDronePosWrapper
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from src.models.ppo_cnn import PpoCnnModel
from src.models.cnn_config import CNNConfig
from src.utils.random_position_wrapper import RandomPositionWrapper


def env_creator(params, args):
    print("-------------------------- ENV CREATOR --------------------------")
    env = CoverageDroneSwarmSearch(
        timestep_limit=200,
        drone_amount=args.n_agents,
        prob_matrix_path=args.matrix_path,
    )

    grid_size = env.grid_size
    print("Grid size: ", grid_size)

    if args.use_random_positions:
        env = RandomPositionWrapper(env)
    else:
        env = RetainDronePosWrapper(env, position_on_edges(grid_size, args.n_agents))

    return env


def position_on_edges(grid_size, n_agents):
    positions = [
        (0, grid_size // 2),
        (grid_size - 1, grid_size // 2),
        (grid_size // 2, 0),
        (grid_size // 2, grid_size - 1),
    ]
    return positions[0:n_agents]


def main(args):
    ray.init()

    env_name = "DSSE_Coverage"

    register_env(
        env_name, lambda config: ParallelPettingZooEnv(env_creator(config, args))
    )
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

    tune.run(
        "PPO",
        name="PPO_" + args.exp_name,
        # resume=True,
        stop={"timesteps_total": 20_000_000},
        checkpoint_freq=20,
        storage_path=args.storage_path + env_name,
        config=config.to_dict(),
    )
