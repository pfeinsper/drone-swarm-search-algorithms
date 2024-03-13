from dataclasses import dataclass


@dataclass
class DQNHyperparameters:
    max_episodes: int = 100_000
    max_steps: int = 1000
    learning_rate: float = 0.001
    gamma: float = 0.99
    epsilon: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.995
    batch_size: int = 64
    update_target_every: int = 100
    memory_size: int = 2000
