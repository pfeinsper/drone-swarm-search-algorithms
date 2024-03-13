import random
import torch
import numpy as np
from collections import deque
from .replay_memory import ReplayMemory, Transition

NUM_TOP_POSITIONS = 10


class DQNAgent:
    def __init__(self, state_size, action_size, hyperparameters, index):
        self.n_actions = action_size
        # we define some parameters and hyperparameters:
        # "lr" : learning rate
        # "gamma": discounted factor
        # "exploration_proba_decay": decay of the exploration probability
        # "batch_size": size of experiences we sample to train the DNN
        self.lr = hyperparameters.learning_rate
        self.gamma = hyperparameters.gamma
        self.episilon = hyperparameters.epsilon
        self.epsilon_decay = hyperparameters.epsilon_decay
        self.epsilon_min = hyperparameters.epsilon_min
        self.batch_size = 32
        self.index = index
        self.name = f"drone{index}"

        # NN setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        self.num_actions = action_size
        # state size being the number of drones, we add the number positions of the probability matrix
        # to consider in the input and multiply by 2 to consider x and y coordinates
        self.num_entries = (state_size + NUM_TOP_POSITIONS) * 2

        self.nn = self.create_nn(self.num_entries, self.num_actions).to(self.device)
        
        self.optimizer = self.create_optimizer(self.nn.parameters())

        # We define our memory buffer where we will store our experiences
        self.memory_buffer = ReplayMemory(hyperparameters.memory_size)

    def create_nn(self, input_dim, output_dim):
        dtype = torch.float
        model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 512, device=self.device, dtype=dtype),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256, device=self.device, dtype=dtype),
            torch.nn.ReLU(),
            torch.nn.Linear(256, output_dim, device=self.device, dtype=dtype),
            torch.nn.Softmax(dim=-1),
        )
        return model.float()

    def predict(self, state):
        nn_input = state.float().to(self.device)
        return self.nn(nn_input)

    def create_optimizer(self, parameters):
        return torch.optim.Adam(parameters, lr=self.lr)

    def select_action(self, current_state):
        # Use epsilon-greedy strategy to choose the next action
        if np.random.uniform(0, 1) < self.episilon:
            return np.random.choice(range(self.n_actions))

        q_values = self.predict(current_state)
        return torch.argmax(q_values)

    def update_exploration_probability(self):
        if self.episilon > self.epsilon_min:
            self.episilon *= self.epsilon_decay

    def store_episode(self, current_state, action, reward, next_state, done):
        self.memory_buffer.push(current_state, action, next_state, reward, done)

    def train(self):
        # We shuffle the memory buffer and select a batch size of experiences
        batch_sample = self.memory_buffer.sample(self.batch_size)


        for experience in batch_sample:
            # We compute the Q-values of S_t
            q_current_state = self.predict(experience["current_state"])
            q_current_state = q_current_state.cpu().detach().numpy()
            # We compute the Q-target using Bellman optimality equation
            q_target = experience["reward"]
            if not experience["done"]:
                q_target = q_target + self.gamma * torch.max(
                    self.predict(experience["next_state"])
                )
            # Update the Q-value of the action taken
            q_current_state[experience["action"]] = q_target
            # train the model
            self.fit(experience["current_state"], q_current_state)

    def fit(self, state, q_values):
        self.optimizer.zero_grad()
        pred = self.predict(state)
        q_values = torch.tensor(q_values, device=self.device)
        loss = torch.nn.functional.mse_loss(pred, q_values)
        loss.backward()
        self.optimizer.step()
    

    def save_model(self, path):
        torch.save(self.nn, path)

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
        ).to(self.device)

        return res