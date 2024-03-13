from DSSE import DroneSwarmSearch

from config import get_config
from algorithms.greedy_search import policy, get_random_vector


def run_one_episode_greedy(env, config=1):
    observation = env.reset(vector=get_random_vector())
    steps_count = 0
    total_reward = 0
    done = False

    while not done:
        actions = policy(observation, env.get_agents())

        observation, reward_dict, _, done, info = env.step(actions)
        done = any(done.values())

        total_reward += reward_dict["total_reward"]
        steps_count += 1

    print(observation)
    # file.write(f"{total_reward}, {steps_count}, {info['Found']}\n")

    return total_reward, steps_count, info["Found"]


def test_greedy_n_times(n_times=100, config_number=0):
    config = get_config(config_number)

    env = DroneSwarmSearch(
        grid_size=config.grid_size,
        render_mode="human",
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

    file_path = f"data/results_greedy_{config.grid_size}_a{config.n_drones}_dc{config.disperse_constant}.csv"
    with open(file_path, "w+", encoding="utf-8") as file:
        file.write("total_reward, steps_count, found\n")
        for _ in range(n_times):
            reward, steps, f = run_one_episode_greedy(env, file)
            total_reward += reward
            steps_count += steps
            found += f

    print(f"Average reward: {total_reward / n_times}")
    print(f"Average steps: {steps_count / n_times}")
    print(f"Found: {found / n_times * 100}% of the time")

    return total_reward / 100, steps_count / 100, found / 100


if __name__ == "__main__":
    test_greedy_n_times(config_number=4, n_times=100)
