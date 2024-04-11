from dataclasses import dataclass


@dataclass
class EnvConfig:
    """Configuration for environment variables"""

    grid_size: int
    drone_amount: int
    vector: list[float]
    drones_initial_positions: list[list[float]]
    person_initial_position: list[float]
    dispersion_inc: float
    timestep_limit: int


def get_config(config_number: int) -> EnvConfig:
    """Configuration for environment variables"""

    env_config = EnvConfig(
        grid_size=40,
        render_mode="human",
        render_grid=True,
        render_gradient=True,
        vector=(3.2, 3.1),
        timestep_limit=200,
        person_amount=2,
        dispersion_inc=0.1,
        person_initial_position=(10, 10),
        drone_amount=1,
        drone_speed=10,
        probability_of_detection=0.9,
        pre_render_time=0,
    )

    match config_number:
        case 1:
            env_config.dispersion_inc = 1
            env_config.drone_amount = 1
        case 2:
            env_config.dispersion_inc = 1
            env_config.drone_amount = 4
        case 3:
            env_config.dispersion_inc = 5
            env_config.drone_amount = 1
        case 4:
            env_config.dispersion_inc = 5
            env_config.drone_amount = 4
    return env_config
