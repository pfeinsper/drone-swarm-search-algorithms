from .recorder import PygameRecord
import numpy as np
import pandas as pd


def print_mean(values, name):
    print(f"Mean of {name}: ", sum(values) / len(values))


def play_with_record(env, agent):
    reward_sum = 0
    obs, info = env.reset()
    with PygameRecord("test_trained.gif", 5) as rec:
        while env.agents:
            actions = {}
            for k, v in obs.items():
                actions[k] = agent.compute_single_action(v, explore=False)
            obs, rw, _, _, info = env.step(actions)
            reward_sum += sum(rw.values())
            rec.add_frame()
    print(info)
    print(reward_sum)


def evaluate_agent_search(env, agent, n_evals=5000):
    rewards = []
    actions_stat = []
    founds = 0
    for _ in range(n_evals):
        obs, info = env.reset()
        actions = 0
        reward_sum = 0
        while env.agents:
            actions = {}
            for k, v in obs.items():
                actions[k] = agent.compute_single_action(v, explore=False)
            obs, rw, _, _, info = env.step(actions)
            reward_sum += sum(rw.values())
            actions += 1
        rewards.append(reward_sum)
        actions_stat.append(actions)
        founds += any(info.values(), key=lambda x: x["Found"])

    print_mean(rewards, "rewards")
    print_mean(actions_stat, "steps needed")
    print("Median of actions: ", np.median(actions_stat))
    print("Found %: ", founds / n_evals)


def evaluate_agent_coverage(env, agent, n_evals=5):
    rewards = []
    cov_rate = []
    steps_needed = []
    repeated_cov = []
    cumulative_metrics = pd.DataFrame()

    for _ in range(n_evals):
        steps = 0
        reward_sum = 0
        cumm_pos = []
        dynamic_cov_rate = []
        dynamic_repeated_cov = []

        obs, info = env.reset()
        while env.agents:
            actions = {}
            for k, v in obs.items():
                actions[k] = agent.compute_single_action(v, explore=False)
            obs, rw, _, _, info = env.step(actions)
            reward_sum += sum(rw.values())
            steps += 1
            cumm_pos.append(info["drone0"]["accumulated_pos"])
            dynamic_cov_rate.append(info["drone0"]["coverage_rate"])
            dynamic_repeated_cov.append(info["drone0"]["repeated_coverage"])

        cumulative_metrics = pd.concat(
            [
                cumulative_metrics,
                pd.DataFrame(
                    {
                        "accumulated_pos": cumm_pos,
                        "step": range(len(cumm_pos)),
                        "coverage_rate": dynamic_cov_rate,
                        "repeated_coverage": dynamic_repeated_cov,
                        "algorithm": "PPO",
                    }
                ),
            ],
            ignore_index=True,
        )
        rewards.append(reward_sum)
        steps_needed.append(steps)
        cov_rate.append(info["drone0"]["coverage_rate"])
        repeated_cov.append(info["drone0"]["repeated_coverage"])

    print_mean(rewards, "rewards")
    print_mean(steps_needed, "steps needed")
    print_mean(cov_rate, "coverage rate")
    print_mean(repeated_cov, "repeated coverage")
    df = pd.DataFrame(
        {
            "rewards": rewards,
            "steps_needed": steps_needed,
            "coverage_rate": cov_rate,
            "repeated_coverage": repeated_cov,
        }
    )
    df.to_csv("coverage_results.csv", index=False)
    cumulative_metrics.to_csv("cumm_metrics.csv", index=False)

    print(f"Total reward: {reward_sum}")
    print(f"Total steps: {steps}")
    print("Info: ", info)
    env.close()
