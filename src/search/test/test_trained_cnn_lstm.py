from DSSE import DroneSwarmSearch
from DSSE.environment.wrappers import RetainDronePosWrapper, AllPositionsWrapper
import ray
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPO
from models import PpoCnnLstmModel
from src.utils.play_env import play_with_record, evaluate_agent_search


def main(args):
    def env_creator(_):
        """
        Petting Zoo environment for search of shipwrecked people.
        check it out at
            https://github.com/pfeinsper/drone-swarm-search
        or install with
            pip install DSSE
        """
        render_mode = "human" if args.see else "ansi"
        env = DroneSwarmSearch(
            drone_amount=4,
            grid_size=40,
            dispersion_inc=0.1,
            person_initial_position=(20, 20),
            render_mode=render_mode,
            render_grid=True,
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


    env = env_creator(None)
    register_env("DSSE", lambda config: ParallelPettingZooEnv(env_creator(config)))
    model = PpoCnnLstmModel
    ModelCatalog.register_custom_model(model.NAME, model)
    ray.init()


    checkpoint_path = args.checkpoint
    PPOagent = PPO.from_checkpoint(checkpoint_path)

    if args.see:
        play_with_record(env, PPOagent)
    else:
        evaluate_agent_search(env, PPOagent)

    env.close()
