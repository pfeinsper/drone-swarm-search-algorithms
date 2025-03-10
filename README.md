[![PyPI Release 🚀](https://badge.fury.io/py/DSSE.svg)](https://badge.fury.io/py/DSSE)
[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg?style=flat)](https://github.com/pfeinsper/drone-swarm-search/blob/main/LICENSE)
[![PettingZoo version dependency](https://img.shields.io/badge/PettingZoo-v1.22.3-blue)]()
![GitHub stars](https://img.shields.io/github/stars/pfeinsper/drone-swarm-search-algorithms)

# <img src="https://github.com/pfeinsper/drone-swarm-search-algorithms/blob/main/imgs/drone.svg" alt="DSSE Icon" width="45" height="25"> Algorithms for Drone Swarm Search (DSSE)

Welcome to the official GitHub repository for the Drone Swarm Search (DSSE) algorithms. These algorithms are specifically tailored for reinforcement learning environments aimed at optimizing drone swarm coordination and search efficiency.

Explore a diverse range of implementations that leverage the latest advancements in machine learning to solve complex coordination tasks in dynamic and unpredictable environments.

## How to run

Arguments inside [brackets] are optional.

#### Run a training script

```sh
python run_script --file <training_file> [--exp_name <name of your experiment>] [--n_agents <number of agents>]
```

Example: training a MLP on the coverage environment.
```sh
python run_script --file train_ppo_mlp_cov.py --exp_name training_mlp_cov
```


#### Run a test script

```sh
python run_script.py --file <test script name> --checkpoint <path to checkpoint to evaluate> [--matrix_path <path to matrix if coverage env>] [--see] [--n_agents <number of agents>]
```

The --see switch makes shows you the agents playing and records a GIF on the env instead of collecting metrics.

Example: Evaluating 4 agents trained on coverage env
```sh
python run_script.py --file test_trained_cov_mlp.py --checkpoint <my_checkpoint_path> --matrix_path data/min_matrix.npy --n_agents 4
```


#### Using the docker container

```
docker compose run --rm dsse-algorithms
```

## 📚 Documentation Links

- **[Documentation Site](https://pfeinsper.github.io/drone-swarm-search/)**: Access detailed tutorials, usage examples, and comprehensive technical documentation. This resource is essential for understanding the DSSE framework and integrating these algorithms into your projects effectively.

- **[DSSE Training Environment Repository](https://github.com/pfeinsper/drone-swarm-search)**: Visit the repository for the DSSE training environment, where you can access the core environment setups and configurations used for developing and testing the algorithms.

- **[PyPI Repository](https://pypi.org/project/DSSE/)**: Download the latest release of DSSE, view the version history, and find installation instructions. Keep up with the latest updates and improvements to the algorithms.

## 🆘 Support and Community

Run into a snag? Have a suggestion? Join our community on GitHub! Submit your queries, report bugs, or contribute to discussions by visiting our [issues page](https://github.com/pfeinsper/drone-swarm-search-algorithms/issues). Your input helps us improve and evolve.
