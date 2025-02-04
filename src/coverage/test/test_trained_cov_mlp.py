from DSSE import CoverageDroneSwarmSearch
from DSSE.environment.wrappers import RetainDronePosWrapper, AllFlattenWrapper
import ray
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPO
from src.utils.play_env import play_with_record, evaluate_agent_coverage


def main(args):
    def env_creator(_):
        print("-------------------------- ENV CREATOR --------------------------")
        N_AGENTS = 2
        render_mode = "human" if args.see else "ansi"
        # 6 hours of simulation, 600 radius
        env = CoverageDroneSwarmSearch(
            timestep_limit=200,
            drone_amount=N_AGENTS,
            prob_matrix_path="min_matrix.npy",
            render_mode=render_mode,
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
    main()
