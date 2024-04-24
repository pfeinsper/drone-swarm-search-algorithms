import logging
import numpy as np
import csv

def config_log(results_folder):
    logging.basicConfig(
        filename=f"{results_folder}/training_log.log",
        level=logging.INFO,
        format="[%(levelname)s] %(asctime)s - %(message)s",
        filemode="w",
    )

def log_episode_stats(episode_num, show_actions, show_rewards):
    actions_mean = sum(show_actions) / len(show_actions)
    rewards_mean = sum(show_rewards) / len(show_rewards)
    logging.info(
        "Episode = %s, Actions (mean) = %s, Reward (mean) = %s",
        episode_num,
        actions_mean,
        rewards_mean,
    )

def get_random_speed_vector():
    return (
        round(np.random.uniform(-1.1, 1.1), 1),
        round(np.random.uniform(-1.1, 1.1), 1),
    )

def init_results_writer(results_folder):
    results_file_name = results_folder + "/results.csv"
    results_file = open(results_file_name, "w", newline="")
    writer = csv.writer(results_file)
    writer.writerow(["episode", "actions", "rewards"])
    return writer, results_file