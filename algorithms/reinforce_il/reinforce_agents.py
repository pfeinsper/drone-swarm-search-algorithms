import os
import torch
import numpy as np
import logging
from .reinforce_agent import ReinforceAgent

NUM_TOP_POSITIONS = 10


class ReinforceAgentsIL:
    def __init__(self, env, gamma, lr, episodes, drones_initial_positions):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_agents = len(env.possible_agents)

        self.gamma = gamma
        self.lr = lr
        self.episodes = episodes
        self.drones_initial_positions = drones_initial_positions

        self.num_agents = len(env.possible_agents)
        self.num_entries = (self.num_agents + NUM_TOP_POSITIONS) * 2
        self.num_actions = len(env.action_space("drone0"))

        self.agents: dict[str, ReinforceAgent] = {}
        for index in range(self.num_agents):
            drone_name = f"drone{index}"
            self.agents[drone_name] = ReinforceAgent(
                index, self.device, self.num_actions, self.num_entries, self.lr
            )

        logging.basicConfig(
            filename=f"logs/training_{self.env.grid_size}_{self.num_agents}_{self.env.disperse_constant}_reinforce_il.log",
            level=logging.INFO,
            format="[%(levelname)s] %(asctime)s - %(message)s",
            filemode="w",
        )

    def log_episode_stats(self, episode_num, show_actions, show_rewards):
        actions_mean = sum(show_actions) / len(show_actions)
        rewards_mean = sum(show_rewards) / len(show_rewards)
        logging.info(
            "Episode = %s, Actions (mean) = %s, Reward (mean) = %s",
            episode_num,
            actions_mean,
            rewards_mean,
        )

    def get_random_speed_vector(self):
        return [
            round(np.random.uniform(-0.1, 0.1), 1),
            round(np.random.uniform(-0.1, 0.1), 1),
        ]

    def train(self):
        statistics, show_rewards, show_actions, all_rewards = [], [], [], []
        stop = False

        for i in range(self.episodes + 1):
            if stop:
                break

            vector = self.get_random_speed_vector()
            observations = self.env.reset(vector=vector)
            done = False
            actions, states, rewards = [], [], []
            count_actions, total_reward = 0, 0

            while not done:
                episode_actions, obs_list = self.select_actions(observations)
                observations, reward_dict, _, done, _ = self.env.step(episode_actions)

                actions.append(
                    torch.tensor(list(episode_actions.values()), dtype=torch.int)
                )
                states.append(obs_list)
                rewards.append(self.extract_rewards(reward_dict))
                count_actions += self.num_agents
                total_reward += reward_dict["total_reward"]
                done = any(done.values())

            show_rewards.append(total_reward)
            all_rewards.append(total_reward)
            show_actions.append(count_actions)

            if len(all_rewards) > 100:
                if all(r >= 100_000 for r in all_rewards[-80:]):
                    stop = True
                    logging.info("[INFO] Early stopping")

            if i % 100 == 0:
                self.log_episode_stats(i, show_actions, show_rewards)
                show_rewards, show_actions = [], []
            if i % 5_000 == 0:
                self.save_neural_networks("checkpoints/")

            statistics.append([i, count_actions, total_reward])
            discounted_returns = self.calculate_discounted_returns(rewards)
            self.update_neural_networks(states, actions, discounted_returns)

        self.save_neural_networks("models/")
        return statistics

    def extract_rewards(self, reward_dict):
        # Get the rewards of the drones exclude "total_reward".
        rewards = [
            drone_reward for key, drone_reward in reward_dict.items() if "drone" in key
        ]
        return rewards

    def select_actions(self, obs_list):
        actions = {}
        agent_inputs = []
        for index in range(self.num_agents):
            drone_name = f"drone{index}"
            agent = self.agents[f"drone{index}"]

            agent_input = agent.flatten_state(obs_list)
            agent_inputs.append(agent_input)

            actions[drone_name] = agent.select_action(agent_input)

        return actions, agent_inputs

    def calculate_discounted_returns(self, rewards):
        discounted_returns = []
        for t in range(len(rewards)):
            G_list = []
            for drone_index in range(self.num_agents):
                agent_rewards = [r[drone_index] for r in rewards]
                G_list.append(
                    sum((self.gamma**k) * r for k, r in enumerate(agent_rewards[t:]))
                )
            discounted_returns.append(G_list)

        return discounted_returns

    def save_neural_networks(self, path):
        # Check if the path exists, if not, create it
        if not os.path.exists(path):
            os.makedirs(path)

        for index in range(self.num_agents):
            torch.save(
                self.agents[f"drone{index}"].nn,
                f"{path}nn_{self.env.grid_size}_{self.num_agents}_{self.env.disperse_constant}_drone{index}_reinforce_il.pt",
            )

    def update_neural_networks(self, states, actions, G_list):
        for state_list, action_list, g in zip(states, actions, G_list):
            for index in range(self.num_agents):
                self.agents[f"drone{index}"].update_neural_network(
                    state_list, action_list, g
                )

    def __call__(self, observations: dict, agents: list) -> dict[str, int]:
        return self.select_actions(observations)[0]

    @classmethod
    def from_trained(cls, env, config):
        agents = {}
        num_agents = len(env.get_agents())

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for index in range(num_agents):
            name = f"drone{index}"
            agents[name] = ReinforceAgent.load_from_file(
                path=f"models/nn_{config}_{name}_reinforce_il.pt",
                device=device,
                num_actions=len(env.action_space(name)),
                num_entries=(num_agents + NUM_TOP_POSITIONS) * 2,
                index=index,
            )
        instance = cls(env, 0.9, 0, 0, [])
        instance.agents = agents
        return instance

    def get_algorithm_name(self):
        return "reinforce_il"