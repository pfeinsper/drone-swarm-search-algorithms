import pathlib
from DSSE import DroneSwarmSearch
from DSSE.environment.wrappers import RetainDronePosWrapper, AllPositionsWrapper
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from models import PpoCnnLstmModel


def env_creator(args):
    """
    Petting Zoo environment for search of shipwrecked people.
    check it out at
        https://github.com/pfeinsper/drone-swarm-search
    or install with
        pip install DSSE
    """
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

    model = PpoCnnLstmModel
    register_env(env_name, lambda config: ParallelPettingZooEnv(env_creator(config)))
    ModelCatalog.register_custom_model(model.NAME, model)

    config = (
        PPOConfig()
        .environment(env=env_name)
        .rollouts(num_rollout_workers=6, rollout_fragment_length="auto")
        .training(
            train_batch_size=4096,
            lr=1e-5,
            gamma=0.9999999,
            lambda_=0.9,
            use_gae=True,
            entropy_coeff=0.01,
            sgd_minibatch_size=300,
            num_sgd_iter=10,
            model={
                "custom_model": "CNNModel",
                "use_lstm": False,
                "lstm_cell_size": 256,
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
        name="PPO_LSTM_M",
        resume=True,
        stop={"timesteps_total": 20_000_000, "episode_reward_mean": 1.82},
        checkpoint_freq=15,
        storage_path=f"{curr_path}/ray_res/" + env_name,
        config=config.to_dict(),
    )
