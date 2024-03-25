import torch
import torch.optim as optim
import torch.nn as nn
from .replay_memory import ReplayMemory, Transition
import random
import math
from .model import DQN
import numpy as np
from .dqn_hyperparameters import DQNHyperparameters

NUM_TOP_POSITIONS = 10

class DQNAgent:
    def __init__(self, state_size: int, action_size: int, params: DQNHyperparameters, index: int) -> None:
        self.index = index
        self.name = f"drone{index}"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        n_observations = (state_size + NUM_TOP_POSITIONS) * 2
        self.num_actions = action_size

        lr = params.learning_rate
        self.tau = params.tau
        self.gamma = params.gamma
        self.batch_size = params.batch_size
        self.episodes = params.max_episodes
        # Epsilon greedy parameters
        self.epsilon_start = params.epsilon
        self.epsilon_min = params.epsilon_min
        self.epsilon_dec = params.epsilon_decay

        policy_net = DQN(n_observations, action_size).to(device)
        target_net = DQN(n_observations, action_size).to(device)
        target_net.load_state_dict(policy_net.state_dict())
        self.policy_net = policy_net
        self.target_net = target_net

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr, amsgrad=True)
        self.memory = ReplayMemory(params.memory_size)
        self.steps_done = 0

    def select_action(self, state):
        sample = random.random()

        eps_threshold = self.epsilon_min + (
            self.epsilon_start - self.epsilon_min
        ) * math.exp(-1.0 * self.steps_done / self.epsilon_dec)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor(
                [[random.randrange(0, self.num_actions)]], device=self.device, dtype=torch.long
            )
            

    def optimize_model(self):
        if len(self.memory) < self.batch_size or self.steps_done % 4 != 0:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device,
            dtype=torch.bool,
        )
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        )
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = (
                self.target_net(non_final_next_states).max(1).values
            )
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def update_target(self):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[
                key
            ] * self.tau + target_net_state_dict[key] * (1 - self.tau)
        self.target_net.load_state_dict(target_net_state_dict)

    def store_episode(self, state, action, reward, next_state):
        reward = torch.tensor([reward], device=self.device)
        self.memory.push(state, action, next_state, reward)
    
    def save_model(self, path):
        torch.save(self.policy_net, path)

    def get_flatten_top_probabilities_positions(self, probability_matrix):
        flattened_probs = probability_matrix.flatten()
        indices = flattened_probs.argsort()[-NUM_TOP_POSITIONS:][::-1]
        positions = np.unravel_index(indices, probability_matrix.shape)
        positions = np.stack((positions[1], positions[0]), axis=-1)
        return positions.flatten()

    def flatten_state(self, observations):
        drone_position = torch.tensor(
            observations[self.name]["observation"][0],
            device=self.device,
        )
        others_position = torch.flatten(
            torch.tensor(
                [
                    observations[drone_idx]["observation"][0]
                    for drone_idx in observations
                    if drone_idx != self.name
                ],
                device=self.device,
            )
        )
        flatten_top_probabilities = torch.tensor(
            self.get_flatten_top_probabilities_positions(
                observations[self.name]["observation"][1]
            ),
            device=self.device,
        )
        res = torch.cat(
            (drone_position, others_position, flatten_top_probabilities), dim=-1
        ).float().to(self.device)
        return res.unsqueeze(0)


    @classmethod
    def load_from_file(cls, path, device, num_actions, num_entries, index):
        print(f"Loading model from {path}")
        model = torch.load(path)
        agent = cls(num_entries, num_actions, DQNHyperparameters(), index)
        agent.policy_net = model.to(device)
        return agent