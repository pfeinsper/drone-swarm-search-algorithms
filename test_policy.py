import argparse
from DSSE import DroneSwarmSearch
from config import get_config
import numpy as np
from algorithms import ReinforceAgent, ReinforceAgentsIL, DQNAgents, greedy_policy

IMPLEMENTED_POLICIES = ["reinforce", "reinforce_il", "ddqn", "greedy"]


def parse_args():
    parser = argparse.ArgumentParser(description="Reinforce")
    parser.add_argument(
        "--config_number",
        type=int,
        default=1,
        help="Configuration number to run the algorithm",
        choices=range(1, 5),
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="reinforce",
        help="Policy (model) to use",
        choices=IMPLEMENTED_POLICIES,
    )
    parser.add_argument(
        "--statistics",
        action="store_true",
        help="If True, will run 100 episodes and print statistics",
    )
    return parser.parse_args()


def play_one_episode(env, policy):
    observation = env.reset(vector=get_random_speed_vector())
    steps_count = 0
    total_reward = 0
    done = False

    while not done:
        actions = policy(observation, env.get_agents())

        observation, reward_dict, _, done, info = env.step(actions)
        done = any(done.values())

        total_reward += reward_dict["total_reward"]
        steps_count += 1

    return total_reward, steps_count, info["Found"]

def get_random_speed_vector():
    return [
        round(np.random.uniform(-0.1, 0.1), 1),
        round(np.random.uniform(-0.1, 0.1), 1),
    ]

def get_model(policy_name, env_to_use: DroneSwarmSearch, config):
    match policy_name:
        case "reinforce":
            return ReinforceAgent.from_trained(env_to_use, config)
        case "reinforce_il":
            return ReinforceAgentsIL.from_trained(env_to_use, config)
        case "ddqn":
            raise NotImplementedError("DQNAgents not implemented yet")
        case "greedy":
            return greedy_policy
        case _:
            raise ValueError(f"Policy {policy_name} not implemented")


def test_n_times(env, model, n=100) -> tuple[float, float, float]:
    total_reward = 0
    steps_count = 0
    found = 0

    for _ in range(n):
        reward, steps, f = play_one_episode(env, model)
        total_reward += reward
        steps_count += steps
        found += f

    return total_reward / n, steps_count / n, found / n

if __name__ == "__main__":
    args = parse_args()
    config = get_config(args.config_number)
    
    RENDER_MODE = "human" if not args.statistics else "ansi"

    env = DroneSwarmSearch(
        grid_size=config.grid_size,
        render_mode=RENDER_MODE,
        render_grid=True,
        n_drones=config.n_drones,
        vector=config.vector,
        person_initial_position=config.person_initial_position,
        disperse_constant=config.disperse_constant,
        timestep_limit=config.timestep_limit,
    )
    print(f"Using config: {config}")
    model = get_model(args.policy, env, config)

    if args.statistics:
        average_reward, average_steps, found_percentage = test_n_times(env, model, n=100)
        print(f"Average reward: {average_reward}")
        print(f"Average steps: {average_steps}")
        print(f"Found: {found_percentage * 100}% of the time")
    else:
        total_reward, steps_count, found = play_one_episode(env, model)
        print(f"Total reward: {total_reward}, Steps: {steps_count}, Found: {found}")
