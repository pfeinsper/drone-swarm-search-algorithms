from .recorder import PygameRecord
import numpy as np


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


def evaluate_agent_coverage(env, agent, n_evals=500):
    rewards = []
    cov_rate = []
    steps_needed = []
    repeated_cov = []

    for _ in range(n_evals):
        i = 0
        obs, info = env.reset()
        reward_sum = 0
        while env.agents:
            actions = {}
            for k, v in obs.items():
                actions[k] = agent.compute_single_action(v, explore=False)
            obs, rw, _, _, info = env.step(actions)
            reward_sum += sum(rw.values())
            i += 1
        rewards.append(reward_sum)
        steps_needed.append(i)
        cov_rate.append(info["drone0"]["coverage_rate"])
        repeated_cov.append(info["drone0"]["repeated_coverage"])

    print_mean(rewards, "rewards")
    print_mean(steps_needed, "steps needed")
    print_mean(cov_rate, "coverage rate")
    print_mean(repeated_cov, "repeated coverage")

    print(f"Total reward: {reward_sum}")
    print(f"Total steps: {i}")
    print("Info: ", info)
    env.close()
