import logging
import numpy as np
from .dqn_hyperparameters import DQNHyperparameters
from .dqn_agent import DQNAgent


class DQNAgents:
    def __init__(self, env, hyperparameters: DQNHyperparameters, config) -> None:
        self.env = env
        self.n_epochs = hyperparameters.max_episodes
        n_agents = len(env.get_agents())
        self.agents = [
            DQNAgent(n_agents, len(env.action_space("drone0")), hyperparameters, index)
            for index in range(n_agents)
        ]

        logging.basicConfig(
            filename=f"logs/training_{config}_dqn.log",
            level=logging.INFO,
            format="[%(levelname)s] %(asctime)s - %(message)s",
            filemode="w",
        )
        self.config = config

    def train(self):
        total_steps = 0
        statistics, show_actions, all_rewards = [], [], []
        stop = False
        batch_size = self.agents[0].batch_size

        for epoch in range(self.n_epochs):
            if stop:
                break

            vector = self.get_random_speed_vector()
            curr_state = self.env.reset(vector=vector)
            curr_state = self.transform_state(curr_state)
            done = False
            count_actions = total_reward = 0

            while not done:
                total_steps += 1

                actions, actions_tensors = self.select_action(curr_state)
                next_state, reward_dict, _, done, _ = self.env.step(actions)

                next_state = self.transform_state(next_state)
                done = any(done.values())
                if done:
                    next_state = [None] * self.config.n_drones

                self.store_episode(curr_state, actions_tensors, reward_dict, next_state)

                count_actions += self.config.n_drones
                total_reward += reward_dict["total_reward"]

                curr_state = next_state

            if epoch > 0 and epoch % 100 == 0:
                self.log_episode_stats(epoch, show_actions, all_rewards)
                # print(self.agents[0].episilon)
                show_actions = []
            if epoch % 5_000 == 0:
                self.save_model("checkpoints")

            if len(all_rewards) > 100:
                if all(r >= 100_000 for r in all_rewards[-90:]):
                    stop = True
                    logging.info("Early stopping due to rewards convergence")

            show_actions.append(count_actions)
            all_rewards.append(total_reward)
            statistics.append([epoch, count_actions, total_reward])

            # Must have at least batch_size samples in memory to start training
            if total_steps >= batch_size:
                self.train_nn()

            for agent in self.agents:
                agent.update_exploration_probability()
                agent.update_target_nn()

        self.save_model("models")
        return statistics

    def transform_state(self, state):
        """
        Gets the observation dict and transform on a list of flatten tensors,
        the layout of the tensor is defined in the Agent class.
        """
        return [agent.flatten_state(state) for agent in self.agents]

    def store_episode(self, curr_state, actions, rewards, next_state):
        for agent in self.agents:
            agent_index = agent.index
            agent_name = agent.name
            agent.store_episode(
                current_state=curr_state[agent_index],
                action=actions[agent_index],
                reward=rewards[agent_name],
                next_state=next_state[agent_index],
            )

    def select_action(self, state):
        actions = {}
        actions_tensor_list = []
        for agent in self.agents:
            action = agent.select_action(state[agent.index])
            actions[agent.name] = action.item()
            actions_tensor_list.append(action)
        return actions, actions_tensor_list

    def train_nn(self):
        # Each agent has its memory buffer, thus we just need to .train() each one of them
        for agent in self.agents:
            agent.train()

    def save_model(self, folder):
        for agent in self.agents:
            agent.save_model(f"{folder}/nn_{self.config}_{agent.index}_dqn.pt")

    def get_random_speed_vector(self):
        return [
            round(np.random.uniform(-0.1, 0.1), 1),
            round(np.random.uniform(-0.1, 0.1), 1),
        ]

    def log_episode_stats(self, episode, show_actions, all_rewards):
        last_100_rewards_mean = np.mean(all_rewards[-100:])
        last_100_actions_mean = np.mean(show_actions)
        logging.info(
            "Episode: %s - Mean actions: %s - Mean rewards: %s",
            episode,
            last_100_actions_mean,
            last_100_rewards_mean,
        )
