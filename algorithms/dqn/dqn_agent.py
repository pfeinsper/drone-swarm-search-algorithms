import random
import torch
import numpy as np
from .replay_memory import ReplayMemory, Transition

NUM_TOP_POSITIONS = 10


class DQNAgent:
    def __init__(self, state_size, action_size, hyperparameters, index):
        self.n_actions = action_size
        # Hyperparameters
        self.lr = hyperparameters.learning_rate
        self.gamma = hyperparameters.gamma
        self.episilon = hyperparameters.epsilon
        self.epsilon_decay = hyperparameters.epsilon_decay
        self.epsilon_min = hyperparameters.epsilon_min
        self.tau = hyperparameters.tau
        self.batch_size = hyperparameters.batch_size

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
        # Target NN for double DQN
        self.target_nn = self.create_nn(self.num_entries, self.num_actions).to(
            self.device
        )
        self.target_nn.load_state_dict(self.nn.state_dict())

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
        )
        return model.float()

    def create_optimizer(self, parameters):
        return torch.optim.AdamW(parameters, lr=self.lr, amsgrad=True)

    def select_action(self, current_state):
        # Use epsilon-greedy strategy to choose the next action
        prob = random.random()
        if prob < self.episilon:
            return torch.tensor(
                [[np.random.choice(range(self.n_actions))]],
                device=self.device,
                dtype=torch.long,
            )
        with torch.no_grad():
            return self.predict(current_state).max(1).indices.view(1, 1)

    def predict(self, state):
        nn_input = state.float().to(self.device)
        return self.nn(nn_input)

    def update_exploration_probability(self):
        if self.episilon > self.epsilon_min:
            self.episilon *= self.epsilon_decay

    def store_episode(self, current_state, action, reward, next_state):
        reward = torch.tensor([reward], device=self.device)
        self.memory_buffer.push(
            state=current_state, action=action, next_state=next_state, reward=reward
        )

    def train(self):
        # We shuffle the memory buffer and select a batch size of experiences
        batch_sample = self.memory_buffer.sample(self.batch_size)
        # This converts batch-array of Transitions to Transition of batch-arrays.
        batch_transpose = Transition(*zip(*batch_sample))

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch_transpose.next_state)),
            device=self.device,
            dtype=torch.bool,
        )

        non_final_next_states_list = [
            s for s in batch_transpose.next_state if s is not None
        ]
        state_batch = torch.cat(batch_transpose.state).to(self.device)
        action_batch = torch.cat(batch_transpose.action).to(self.device)
        reward_batch = torch.cat(batch_transpose.reward).to(self.device)

        # We compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
        state_action_values = self.predict(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        # If the state is not final, we use the target_nn to compute the next state values
        if len(non_final_next_states_list) > 0:
            non_final_next_states = torch.cat(non_final_next_states_list).to(
                self.device
            )
            with torch.no_grad():
                nn_inputs = non_final_next_states.float()
                next_state_values[non_final_mask] = (
                    self.target_nn(nn_inputs).max(1).values
                )
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss - like MSE but less sensitive to outliers
        criterion = torch.nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        # torch.nn.utils.clip_grad_value_(self.nn.parameters(), 100)
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

        return res.unsqueeze(0)

    def update_target_nn(self):
        target_net_state_dict = self.target_nn.state_dict()
        policy_net_state_dict = self.nn.state_dict()
        for key in policy_net_state_dict:
            from_current_policy = policy_net_state_dict[key] * self.tau
            from_target_net = target_net_state_dict[key] * (1 - self.tau)
            target_net_state_dict[key] = from_current_policy + from_target_net
        self.target_nn.load_state_dict(target_net_state_dict)
