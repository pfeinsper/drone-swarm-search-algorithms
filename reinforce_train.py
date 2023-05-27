import torch
import pandas as pd
from DSSE.env import DroneSwarmSearch

from config import get_config
from algorithms.reinforce import ReinforceAgent

config = get_config(1)

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

rl_agent = ReinforceAgent(
    env,
    y=0.999999,
    lr=0.000001,
    episodes=100_000,
    drones_initial_positions=config.drones_initial_positions,
)
nn, statistics = rl_agent.train()

torch.save(nn, f"data/nn_{config.grid_size}_{config.grid_size}_{config.n_drones}.pt")
df = pd.DataFrame(statistics, columns=["episode", "actions", "rewards"])
df.to_csv(
    f"data/statistics_{config.grid_size}_{config.grid_size}_{config.n_drones}.csv",
    index=False,
)
