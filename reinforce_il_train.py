import pandas as pd
from DSSE import DroneSwarmSearch

from config import get_config
from algorithms.reinforce_il.reinforce_agents import ReinforceAgentsIL

config = get_config(4)

# Tests:
# N_drones   | 1 | 4 |
# Dispersion | 1 | 5 |

env = DroneSwarmSearch(
    grid_size=config.grid_size,
    render_mode="ansi",
    render_grid=False,
    render_gradient=False,
    n_drones=config.n_drones,
    person_initial_position=config.person_initial_position,
    disperse_constant=config.disperse_constant,
    timestep_limit=config.timestep_limit,
)

rl_agent = ReinforceAgentsIL(
    env,
    gamma=0.999999,
    lr=0.000001,
    episodes=100_000,
    drones_initial_positions=config.drones_initial_positions,
)
statistics = rl_agent.train()

df = pd.DataFrame(statistics, columns=["episode", "actions", "rewards"])
df.to_csv(
    f"data/statistics_{config}.csv",
    index=False,
)
