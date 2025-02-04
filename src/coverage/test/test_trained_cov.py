from DSSE import CoverageDroneSwarmSearch
from DSSE.environment.wrappers import RetainDronePosWrapper, AllPositionsWrapper
import ray
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPO
from src.models.ppo_cnn import PpoCnnModel
from src.models.cnn_config import CNNConfig
from src.utils.play_env import play_with_record, evaluate_agent_coverage


def main(args):
    # Register the model
    model = PpoCnnModel
    model.CONFIG = CNNConfig(kernel_sizes=[(3, 3), (2, 2)])
    ModelCatalog.register_custom_model(model.NAME, model)

    def env_creator(_):
        print("-------------------------- ENV CREATOR --------------------------")
        N_AGENTS = 8
        # 6 hours of simulation, 600 radius
        env = CoverageDroneSwarmSearch(
            timestep_limit=180,
            drone_amount=N_AGENTS,
            prob_matrix_path="presim_20.npy",
            render_mode="human",
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
    register_env(
        "DSSE_Coverage", lambda config: ParallelPettingZooEnv(env_creator(config))
    )
    ray.init()

    checkpoint_path = args.checkpoint
    PPOagent = PPO.from_checkpoint(checkpoint_path)

    if args.see:
        play_with_record(env, PPOagent)
    else:
        evaluate_agent_coverage(env, PPOagent)

    env.close()


if __name__ == "__main__":
    main(None)
