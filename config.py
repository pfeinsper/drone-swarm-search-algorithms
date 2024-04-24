from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class EnvConfig:
    """Configuration for environment variables"""
    grid_size: int
    render_mode: str
    render_grid: bool
    render_gradient: bool
    vector: Tuple[float, float]
    timestep_limit: int
    person_amount: int
    dispersion_inc: float
    person_initial_position: Tuple[float, float]
    drone_amount: int
    drone_speed: int
    probability_of_detection: float
    pre_render_time: int

    def __repr__(self) -> str:
        return f"{self.grid_size}_{self.drone_amount}_{self.dispersion_inc}"

def get_config(config_number: int) -> Tuple[EnvConfig, Dict[str, List[float]]]:
    env_config = EnvConfig(
        grid_size=40,
        render_mode="human",
        render_grid=True,
        render_gradient=True,
        vector=(1, 1),
        timestep_limit=200,
        person_amount=2,
        dispersion_inc=0.1,
        person_initial_position=(10, 10),
        drone_amount=2,
        drone_speed=10,
        probability_of_detection=0.9,
        pre_render_time=0
    )

    match config_number:
        case 1:
            env_config.dispersion_inc = 0.1
            env_config.drone_amount = 4
        case 2:
            env_config.dispersion_inc = 0.01
            env_config.drone_amount = 4
        case 3:
            env_config.dispersion_inc = 0.1
            env_config.drone_amount = 1
        case 4:
            env_config.dispersion_inc = 0.1
            env_config.drone_amount = 4

    return env_config

def get_opt() -> Dict[str, List[float]]:
    return {
    }
