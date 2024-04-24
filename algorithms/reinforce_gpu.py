import logging
import torch
import numpy as np
from DSSE import DroneSwarmSearch
from config import get_opt
from .training_utils import config_log, get_random_speed_vector, log_episode_stats, init_results_writer


NUM_TOP_POSITIONS = 10
class Reinforce:
    def __init__(self, env: DroneSwarmSearch):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_top_positions = 10
        self.num_agents = len(env.possible_agents)

    def get_flatten_top_probabilities_positions(self, probability_matrix):
        flattened_probs = probability_matrix.flatten()
        indices = flattened_probs.argsort()[-self.num_top_positions :][::-1]
        positions = np.unravel_index(indices, probability_matrix.shape)
        # TODO: Discuss -> Before it was (row, col), but drone position is (x, y)
        # Stack the positions in the format (x, y) or (col, row)
        positions = np.stack((positions[1], positions[0]), axis=-1)
        return positions.flatten()

    def flatten_state(self, observations):
        flatten_all = []
        
        for drone_index in range(self.num_agents):
            drone_position = torch.tensor(
                observations["drone" + str(drone_index)][0],
                device=self.device,
            )
            others_position = torch.flatten(
                torch.tensor(
                    [
                        observations["drone" + str(index)][0]
                        for index in range(self.num_agents)
                        if index != drone_index
                    ],
                    device=self.device,
                )
            )
            flatten_top_probabilities = torch.tensor(
                self.get_flatten_top_probabilities_positions(
                    observations["drone" + str(drone_index)][1]
                ),
                device=self.device,
            )
            flatten_all.append(
                torch.cat(
                    (drone_position, others_position, flatten_top_probabilities), dim=-1
                ).to(self.device)
            )

        return flatten_all


class ReinforceAgent(Reinforce):
    def __init__(self, env, y, lr, episodes, drones_initial_positions):
        super().__init__(env)
        self.y = y
        self.lr = lr
        self.episodes = episodes
        self.drones_initial_positions = drones_initial_positions

        self.num_agents = len(env.possible_agents)
        self.num_entries = (self.num_agents + self.num_top_positions) * 2
        self.num_actions = (env.action_space("drone0").n)

        self.nn = self.create_neural_network().to(self.device)
        self.optimizer = self.create_optimizer(self.nn.parameters())
        self.results_folder = None

    def create_neural_network(self):
        dtype = torch.float
        nn = torch.nn.Sequential(
            torch.nn.Linear(self.num_entries, 512, device=self.device, dtype=dtype),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256, device=self.device, dtype=dtype),
            torch.nn.ReLU(),
            torch.nn.Linear(256, self.num_actions, device=self.device, dtype=dtype),
            torch.nn.Softmax(dim=-1),
        )
        return nn.float()

    def create_optimizer(self, parameters):
        optimizer = torch.optim.Adam(parameters, lr=self.lr)
        return optimizer

    def select_actions(self, obs_list):
        episode_actions = {}
        for drone_index in range(self.num_agents):
            probs = self.nn(obs_list[drone_index].float().to(self.device))
            distribution = torch.distributions.Categorical(probs)
            episode_actions[f"drone{drone_index}"] = distribution.sample().item()

        return episode_actions

    def extract_rewards(self, reward_dict):
        rewards = [
            drone_reward for key, drone_reward in reward_dict.items() if "drone" in key
        ]
        return rewards

    def calculate_discounted_returns(self, rewards):
        discounted_returns = []
        for t in range(len(rewards)):
            G_list = []
            for drone_index in range(self.num_agents):
                agent_rewards = [r[drone_index] for r in rewards]
                G_list.append(
                    sum((self.y**k) * r for k, r in enumerate(agent_rewards[t:]))
                )
            discounted_returns.append(G_list)

        return discounted_returns

    def update_neural_network(self, states, actions, discounted_returns):
        for state_list, action_list, G_list in zip(states, actions, discounted_returns):
            for drone_index in range(self.num_agents):
                probs = self.nn(state_list[drone_index].float().to(self.device))
                distribution = torch.distributions.Categorical(probs=probs)
                log_prob = distribution.log_prob(
                    action_list[drone_index].to(self.device)
                )

                loss = -log_prob * G_list[drone_index]

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def save_nn(self, path):
        path = f"{self.results_folder}/{path}"
        torch.save(self.nn, path)

    def train(self, results_folder):
        config_log(results_folder)
        writer, res_file = init_results_writer(results_folder)
        self.results_folder = results_folder
        
        show_rewards, show_actions, all_rewards = [], [], []
        stop = False

        for i in range(self.episodes + 1):
            if stop:
                break

            options = get_opt()
            options["vector"] = get_random_speed_vector()

            observations, _ = self.env.reset(options=options)
            obs_list = self.flatten_state(observations)
            done = False
            actions, states, rewards = [], [], []
            count_actions, total_reward = 0, 0

            while not done:
                episode_actions = self.select_actions(obs_list)
                obs_list_, reward_dict, term, trunc, _ = self.env.step(episode_actions)
                actions.append(
                    torch.tensor(list(episode_actions.values()), dtype=torch.int)
                )
                states.append(obs_list)
                rewards.append(self.extract_rewards(reward_dict))
                obs_list = self.flatten_state(obs_list_)
                count_actions += self.num_agents
                total_reward += sum(reward_dict.values())
                done = any(term.values()) or any(trunc.values())

            show_rewards.append(total_reward)
            all_rewards.append(total_reward)
            show_actions.append(count_actions)

            if len(all_rewards) > 100:
                if all(r >= 100_000 for r in all_rewards[-90:]):
                    stop = True
                    logging.info("Early stopping")

            if i % 100 == 0:
                log_episode_stats(i, show_actions, show_rewards)
                show_rewards, show_actions = [], []
            if i % 5_000 == 0:
                self.save_nn(
                    f"checkpoints/nn_reinforce.pt"
                )

            writer.writerow([i, count_actions, total_reward])
            discounted_returns = self.calculate_discounted_returns(rewards)
            self.update_neural_network(states, actions, discounted_returns)

        self.save_nn(
            f"models/nn_reinforce.pt"
        )
        res_file.close()
    

    @classmethod
    def from_trained(cls, env, config, path=None):
        if not path:
            path = f"models/nn_{config}_reinforce.pt"
        print(f"Loading model from {path}")
        agent = cls(env, 0, 0, 0, [])
        agent.nn = torch.load(path)
        return agent
    
    def __call__(self, observations, _):
        obs_list = self.flatten_state(observations)
        actions = self.select_actions(obs_list)
        return actions

    def __repr__(self):
        return "reinforce"