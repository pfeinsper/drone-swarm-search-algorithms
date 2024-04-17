import numpy as np
import torch
from .net import ConvNetwork


class ReinforceAgent:
    def __init__(self, drone_index, device, num_actions, matrix_shape, n_agents, lr):
        self.drone_index = drone_index
        self.name = f"drone{drone_index}"
        self.device = device

        self.num_actions = num_actions
        self.grid_size = matrix_shape[0]

        self.lr = lr
        self.nn = ConvNetwork(1, matrix_shape, 2 * n_agents, num_actions).to(self.device)
        self.optimizer = self.create_optimizer(self.nn.parameters())

    def flatten_state(self, observations):
        agent_obs = observations[self.name]
        drone_position = torch.tensor(
            agent_obs["observation"][0],
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
        prob_matrix = torch.from_numpy(agent_obs["observation"][1]).unsqueeze(0).float().to(self.device)
        # for pos in others_position:
        #     x, y = pos
        #     prob_matrix[y][x] = -1

        x_scalar = torch.cat((drone_position, others_position), dim=-1)
        # x_scalar = drone_position
        return (x_scalar, prob_matrix)

    def create_optimizer(self, parameters):
        optimizer = torch.optim.Adam(parameters, lr=self.lr)
        return optimizer

    def select_action(self, observations: torch.Tensor):
        probs = self.nn(observations)
        distribution = torch.distributions.Categorical(probs)
        return distribution.sample().item()

    def update_neural_network(self, state_list, action_list, G_list):
        self.nn.train()
        probs = self.nn(state_list[self.drone_index])
        distribution = torch.distributions.Categorical(probs=probs)
        log_prob = distribution.log_prob(action_list[self.drone_index].to(self.device))

        loss = -log_prob * G_list[self.drone_index]

        self.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_value_(self.nn.parameters(), 100)
        self.optimizer.step()
        self.nn.eval()

    @classmethod
    def load_from_file(cls, path, device, num_actions, num_entries, index):
        print(f"Loading neural network from {path}")
        nn = torch.load(path)
        # agent = cls(index, device, num_actions, num_entries, 0)
        agent = ReinforceAgent(index, device, num_actions, (20, 20), 4, 0.001)
        agent.nn = nn
        return agent
