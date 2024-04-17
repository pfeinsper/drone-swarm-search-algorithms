import numpy as np
import torch

NUM_TOP_POSITIONS = 10

class ReinforceAgent:
    def __init__(self, drone_index, device, num_actions, num_entries, lr):
        self.drone_index = drone_index
        self.name = f"drone{drone_index}"
        self.device = device

        self.num_entries = num_entries
        self.num_actions = num_actions

        self.lr = lr
        self.nn = self.create_neural_network().to(self.device)
        self.optimizer = self.create_optimizer(self.nn.parameters())

    def get_flatten_top_probabilities_positions(self, probability_matrix):
        flattened_probs = probability_matrix.flatten()
        indices = flattened_probs.argsort()[-NUM_TOP_POSITIONS:][::-1]
        positions = np.unravel_index(indices, probability_matrix.shape)
        # TODO: Discuss -> Before it was (row, col), but drone position is (x, y)
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
                    observations[index]["observation"][0]
                    for index in observations
                    if index != self.name
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

    def select_action(self, observations: torch.Tensor):
        probs = self.nn(observations.float().to(self.device))
        distribution = torch.distributions.Categorical(probs)
        return distribution.sample().item()

    def update_neural_network(self, state_list, action_list, G_list):
        probs = self.nn(state_list[self.drone_index].float().to(self.device))
        distribution = torch.distributions.Categorical(probs=probs)
        log_prob = distribution.log_prob(action_list[self.drone_index].to(self.device))

        loss = -log_prob * G_list[self.drone_index]

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    @classmethod
    def load_from_file(cls, path, device, num_actions, num_entries, index):
        print(f"Loading neural network from {path}")
        nn = torch.load(path)
        agent = cls(index, device, num_actions, num_entries, 0)
        agent.nn = nn
        return agent