import torch
import pandas as pd
from DSSE.core.environment.env import DroneSwarmSearch

from config import get_config
from algorithms.reinforce_gpu import ReinforceAgent

for config in range(1, 4+1):
    config = get_config(config)

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

    rl_agent = ReinforceAgent(
        env,
        y=0.999999,
        lr=0.000001,
        episodes=100_000,
        drones_initial_positions=config.drones_initial_positions,
    )
    nn, statistics = rl_agent.train()

    torch.save(nn, f"data/nn_{config.grid_size}_{config.n_drones}_{config.disperse_constant}.pt")
    df = pd.DataFrame(statistics, columns=["episode", "actions", "rewards"])
    df.to_csv(
        f"data/statistics_{config.grid_size}_{config.n_drones}_{config.disperse_constant}.csv",
        index=False,
    )
