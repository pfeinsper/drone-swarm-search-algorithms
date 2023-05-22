from algorithms.parallel_sweep import MultipleParallelSweep
from DroneSwarmSearchEnvironment.env import DroneSwarmSearch

from config import get_config


def test_parallel_sweep_100_times(config_number=0):
    config = get_config(config_number)
    env = DroneSwarmSearch(
        grid_size=config.grid_size,
        render_mode="human",
        render_grid=False,
        render_gradient=False,
        n_drones=config.n_drones,
        person_initial_position=config.person_initial_position,
        disperse_constant=config.disperse_constant,
        timestep_limit=200,
    )
    algorithm = MultipleParallelSweep(env)

    total_reward = 0
    steps_count = 0
    found = 0

    for _ in range(100):
        reward, steps, f = algorithm.run()
        total_reward += reward
        steps_count += steps
        found += f

    print(f"Average reward: {total_reward / 100}")
    print(f"Average steps: {steps_count / 100}")
    print(f"Found: {found/100}% of the times")


test_parallel_sweep_100_times()
