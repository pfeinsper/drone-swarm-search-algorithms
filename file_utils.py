import os

EXPERIMENTS_FOLDER = "experiments"


def create_experiment_folder(experiment_name: str) -> str:
    experiment_folder = os.path.join(EXPERIMENTS_FOLDER, experiment_name)
    if os.path.exists(experiment_folder):
        ans = input(
            f"Experiment {experiment_name} already exists. Do you want to overwrite it? (y/n): "
        )
        if ans.lower() != "y":
            exit(0)
    os.makedirs(experiment_folder, exist_ok=True)
    # Create checkpoints and models folder
    os.makedirs(os.path.join(experiment_folder, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(experiment_folder, "models"), exist_ok=True)
    return experiment_folder