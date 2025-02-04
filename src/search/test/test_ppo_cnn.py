from DSSE import DroneSwarmSearch
from DSSE.environment.wrappers import RetainDronePosWrapper, AllPositionsWrapper
import ray
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPO
from models import PpoCnnModel
from src.utils.play_env import play_with_record, evaluate_agent_search


# DEFINE HERE THE EXACT ENVIRONMENT YOU USED TO TRAIN THE AGENT
def env_creator(_):
    render_mode = "human" if args.see else "ansi"
    env = DroneSwarmSearch(
        drone_amount=4,
        grid_size=40,
        render_mode=render_mode,
        render_grid=True,
        dispersion_inc=0.1,
        person_initial_position=(20, 20),
    )
    positions = [
        (20, 0),
        (20, 39),
        (0, 20),
        (39, 20),
    ]
    env = RetainDronePosWrapper(env, positions)
    env = AllPositionsWrapper(env)
    return env

def main(args):
    model = PpoCnnModel
    ModelCatalog.register_custom_model(model.NAME, model)

    env = env_creator(None)
    register_env("DSSE", lambda config: ParallelPettingZooEnv(env_creator(config)))
    ray.init()

    checkpoint_path = args.checkpoint
    PPOagent = PPO.from_checkpoint(checkpoint_path)

    if args.see:
        play_with_record(env, PPOagent)
    else:
        evaluate_agent_search(env, PPOagent)

    env.close()
