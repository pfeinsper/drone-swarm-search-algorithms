from algorithms.parallel_sweep import MultipleParallelSweep
from DroneSwarmSearchEnvironment.env import DroneSwarmSearch

from config import get_config

config = get_config(1)

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


def test_100_times():
    total_success = 0
    for i in range(100):
        success = algorithm.run()

        if success:
            total_success += 1
            print(f"Episode {i} found person")
        else:
            print(f"Episode {i} did not find person")

    print(
        f"Drones found person {total_success} times, {total_success / 100 * 100} % of the time"
    )
    return total_success


test_100_times()
