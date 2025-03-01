import torch.nn as nn
import torch
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from .cnn_config import CNNConfig

class PpoCnnModel(TorchModelV2, nn.Module):
    NAME = "PpoCnnModel"
    CONFIG = CNNConfig()

    def __init__(
        self,
        obs_space,
        act_space,
        num_outputs,
        model_config,
        name,
        **kw,
    ):
        print("OBSSPACE: ", obs_space)
        TorchModelV2.__init__(
            self, obs_space, act_space, num_outputs, model_config, name, **kw
        )
        nn.Module.__init__(self)

        flatten_size = self.CONFIG.get_flatten_size(obs_space[1].shape)
        kernels = self.CONFIG.kernel_sizes
        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=self.CONFIG.out_channels[0],
                kernel_size=kernels[0],
                stride=(1, 1),
            ),
            nn.Tanh(),
            nn.Conv2d(
                in_channels=self.CONFIG.out_channels[0],
                out_channels=self.CONFIG.out_channels[1],
                kernel_size=kernels[1],
                stride=(1, 1),
            ),
            nn.Tanh(),
            nn.Flatten(),
            nn.Linear(flatten_size, 256),
            nn.Tanh(),
        )

        self.linear = nn.Sequential(
            nn.Linear(obs_space[0].shape[0], 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
        )

        self.join = nn.Sequential(
            nn.Linear(256 * 2, 256),
            nn.Tanh(),
        )

        self.policy_fn = nn.Linear(256, num_outputs)
        self.value_fn = nn.Linear(256, 1)

    def forward(self, input_dict, state, seq_lens):
        input_positions = input_dict["obs"][0].float()
        input_matrix = input_dict["obs"][1].float()

        input_matrix = input_matrix.unsqueeze(1)
        cnn_out = self.cnn(input_matrix)
        linear_out = self.linear(input_positions)

        value_input = torch.cat((cnn_out, linear_out), dim=1)
        value_input = self.join(value_input)

        self._value_out = self.value_fn(value_input)
        return self.policy_fn(value_input), state

    def value_function(self):
        return self._value_out.flatten()
