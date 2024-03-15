import argparse
import pandas as pd
from DSSE import DroneSwarmSearch
from algorithms import ReinforceAgent, ReinforceAgentsIL, DQNAgents, DQNHyperparameters
from config import get_config

IMPLEMENTED_MODELS = ["reinforce", "dqn", "reinforce_il"]


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model to solve the DSSE problem.")
    parser.add_argument(
        "--model",
        type=str,
        default="reinforce",
        choices=IMPLEMENTED_MODELS,
        help="The model to train.",
    )
    parser.add_argument(
        "--config",
        type=int,
        default=4,
        help="The configuration to train.",
        choices=range(1, 4 + 1),
    )
    args = parser.parse_args()
    return args


def get_model(model_name, env, config):
    model = None
    match model_name:
        case "reinforce":
            model = ReinforceAgent(
                env,
                y=0.999999,
                lr=0.000001,
                episodes=100_000,
                drones_initial_positions=config.drones_initial_positions,
            )
        case "reinforce_il":
            model = ReinforceAgentsIL(
                env,
                gamma=0.999999,
                lr=0.000001,
                episodes=100_000,
                drones_initial_positions=config.drones_initial_positions,
            )
        case "dqn":
            hyperparameters = DQNHyperparameters(
                max_episodes=100_000,
                learning_rate=0.000001,
                gamma=0.999999,
                epsilon=1.0,
                epsilon_min=0.01,
                epsilon_decay=0.995,
                batch_size=64,
                memory_size=2_000,
            )
            model = DQNAgents(env, hyperparameters, config)
        case _:
            raise NotImplementedError(f"Model {model_name} not implemented.")
    return model


if __name__ == "__main__":
    args = parse_args()
    model_name = args.model
    config = get_config(args.config)
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
    model = get_model(model_name, env, config)

    print(f"Training {model_name} with config {config}...")

    statistics = model.train()
    df = pd.DataFrame(statistics, columns=["episode", "actions", "rewards"])
    df.to_csv(
        f"data/statistics_{config}_{model_name}.csv",
        index=False,
    )
