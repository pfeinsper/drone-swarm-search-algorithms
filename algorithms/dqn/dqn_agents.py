import logging
import torch
from .dqn_hyperparameters import DQNHyperparameters
from .dqn_agent import DQNAgent
from config import get_opt
from ..training_utils import config_log, get_random_speed_vector, log_episode_stats, init_results_writer


class DQNAgents:
    def __init__(self, env, hyperparameters: DQNHyperparameters, config) -> None:
        self.env = env
        self.n_epochs = hyperparameters.max_episodes
        n_agents = len(env.get_agents())
        self.agents = [
            DQNAgent(n_agents, env.action_space("drone0"), hyperparameters, index)
            for index in range(n_agents)
        ]
        self.config = config
        self.results_folder = None

    def train(self, results_folder):
        config_log(results_folder)
        writer, res_file = init_results_writer(results_folder)
        self.results_folder = results_folder
        total_steps = 0
        show_actions, all_rewards = [], []
        stop = False
        batch_size = self.agents[0].batch_size

        for epoch in range(self.n_epochs):
            if stop:
                break
            
            options = get_opt()
            options["vector"] = get_random_speed_vector()
            curr_state, _ = self.env.reset(options=options)
            curr_state = self.transform_state(curr_state)
            done = False
            count_actions = total_reward = 0

            while not done:
                total_steps += 1

                actions, actions_tensors = self.select_action(curr_state)
                next_state, reward_dict, term, trunc, _ = self.env.step(actions)

                next_state = self.transform_state(next_state)
                done = any(term.values()) or any(trunc.values())
                if done:
                    next_state = [None] * self.config.drone_amount

                self.store_episode(curr_state, actions_tensors, reward_dict, next_state)
                if total_steps >= batch_size:
                    self.train_nn()

                count_actions += self.config.drone_amount
                total_reward += sum(reward_dict.values())

                curr_state = next_state

            if epoch > 0 and epoch % 100 == 0:
                log_episode_stats(epoch, show_actions, all_rewards[-100:])
                show_actions = []
            if epoch % 5_000 == 0:
                self.save_model("checkpoints")

            if len(all_rewards) > 100:
                if all(r >= 100_000 for r in all_rewards[-90:]):
                    stop = True
                    logging.info("Early stopping due to rewards convergence")

            show_actions.append(count_actions)
            all_rewards.append(total_reward)
            writer.writerow([epoch, count_actions, total_reward])

            for agent in self.agents:
                agent.update_target()

        self.save_model("models")
        res_file.close()

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
                state=curr_state[agent_index],
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
            agent.optimize_model()

    def save_model(self, folder):
        path = f"{self.results_folder}/{folder}"
        for agent in self.agents:
            agent.save_model(f"{path}/nn_{agent.index}_dqn.pt")

    @classmethod
    def from_trained(cls, env, config, checkpoint=False):
        agents = []
        num_agents = len(env.get_agents())
        folder = "checkpoints" if checkpoint else "models"

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for index in range(num_agents):
            agents.append(DQNAgent.load_from_file(
                path=f"{folder}/nn_{config}_{index}_dqn.pt",
                device=device,
                num_actions=len(env.action_space("drone0")),
                num_entries=num_agents,
                index=index,
            )
        )
        instance = cls(env, DQNHyperparameters(), config)
        instance.agents = agents
        return instance

    def __call__(self, observations, agents_list):
        actions = {}
        state = self.transform_state(observations)
        for agent in self.agents:
            agent_name = agent.name
            with torch.no_grad():
                action = torch.argmax(agent.policy_net(state[agent.index]))
            actions[agent_name] = action.item()
        return actions
    
    def __repr__(self) -> str:
        return "dqn"