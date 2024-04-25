import argparse
from DSSE import DroneSwarmSearch
from algorithms import ReinforceAgent, ReinforceAgentsIL, DQNAgents, DQNHyperparameters
from config import get_opt, get_config
from file_utils import create_experiment_folder

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
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Name of the experiment.",
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
                drones_initial_positions=get_opt(),
            )
        case "reinforce_il":
            model = ReinforceAgentsIL(
                env,
                gamma=0.999999,
                lr=0.000001,
                episodes=100_000,
                drones_initial_positions=get_opt(),
            )
        case "dqn":
            hyperparameters = DQNHyperparameters(
                max_episodes=100_000,
                learning_rate=1e-4,
                gamma=0.999999,
                epsilon=0.9,
                epsilon_min=0.05,
                epsilon_decay=40_000,
                batch_size=256,
                memory_size=20_000,
                tau=0.0005,
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
        vector=config.vector,
        timestep_limit=config.timestep_limit,
        person_amount=config.person_amount,
        dispersion_inc=config.dispersion_inc,
        person_initial_position=config.person_initial_position,
        drone_amount=config.drone_amount,
        drone_speed=config.drone_speed,
        probability_of_detection=config.probability_of_detection,
        pre_render_time=config.pre_render_time,
    )
    model = get_model(model_name, env, config)
    folder = create_experiment_folder(args.name)

    print(f"Training {model_name} with config {config}...")

    model.train(folder)
    
