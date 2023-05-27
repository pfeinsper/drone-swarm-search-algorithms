import torch
from DSSE.env import DroneSwarmSearch

from config import get_config
from algorithms.reinforce import Reinforce


def run_one_episode(env, nn, config):
    reinforce = Reinforce(env=env)
    episode_actions = {}
    state = env.reset(
        drones_positions=config.drones_initial_positions,
        vector=reinforce.get_random_speed_vector(),
    )
    obs_list = reinforce.flatten_state(state)
    steps_count = 0
    total_reward = 0
    done = False

    while not done:
        for drone_index in range(len(env.possible_agents)):
            probs = nn(obs_list[drone_index].float())
            dist = torch.distributions.Categorical(probs)
            episode_actions[f"drone{drone_index}"] = dist.sample().item()

        obs_list_, reward_dict, _, done, info = env.step(episode_actions)
        obs_list = reinforce.flatten_state(obs_list_)
        done = any(done.values())

        total_reward += reward_dict["total_reward"]
        steps_count += 1

    return total_reward, steps_count, info["Found"]


def test_reinforce_100_times(config_number=0):
    config = get_config(config_number)

    nn = torch.load(
        f"data/nn_{config.grid_size}_{config.grid_size}_{config.n_drones}.pt"
    )
    nn = nn.float()

    env = DroneSwarmSearch(
        grid_size=config.grid_size,
        render_mode="ansi",
        render_grid=True,
        render_gradient=True,
        n_drones=config.n_drones,
        vector=config.vector,
        person_initial_position=config.person_initial_position,
        disperse_constant=config.disperse_constant,
        timestep_limit=config.timestep_limit,
    )
    total_reward = 0
    steps_count = 0
    found = 0

    for _ in range(100):
        reward, steps, f = run_one_episode(env, nn, config)
        total_reward += reward
        steps_count += steps
        found += f

    print(f"Average reward: {total_reward / 100}")
    print(f"Average steps: {steps_count / 100}")
    print(f"Found: {found/100 * 100}% of the time")

    return total_reward / 100, steps_count / 100, found / 100


test_reinforce_100_times(1)
