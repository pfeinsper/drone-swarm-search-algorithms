from dataclasses import dataclass


@dataclass
class EnvConfig:
    """Configuration for environment variables"""

    grid_size: int
    n_drones: int
    vector: list[float]
    drones_initial_positions: list[list[float]]
    person_initial_position: list[float]
    disperse_constant: int
    timestep_limit: int

    def __repr__(self) -> str:
        return f"{self.grid_size}_{self.n_drones}_{self.disperse_constant}"


def get_config(config_number: int) -> EnvConfig:
    env_config = EnvConfig(
        grid_size=20,
        n_drones=1,
        vector=[0.1, 0.1],
        drones_initial_positions=None,
        person_initial_position=[10, 10],
        disperse_constant=1,
        timestep_limit=100,
    )

    match config_number:
        case 1:
            env_config.disperse_constant = 1
            env_config.n_drones = 1
        case 2:
            env_config.disperse_constant = 1
            env_config.n_drones = 4
        case 3:
            env_config.disperse_constant = 5
            env_config.n_drones = 1
        case 4:
            env_config.disperse_constant = 5
            env_config.n_drones = 4
    return env_config
