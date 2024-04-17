import torch
import torch.nn as nn
from torch.nn.functional import relu, leaky_relu

# TODO: Should i remove dropout ??

class ConvNetwork(nn.Module):
    def __init__(self, input_channels, matrix_shape, num_scalars, num_classes):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Convolutional layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        # Fully connected layers
        self.fc1_input_size = (
            32 * (matrix_shape[0] // 4) * (matrix_shape[1] // 4)
        )
        # Apply a DENSE layer to the flattened CNN2 output
        self.fc1 = nn.Linear(self.fc1_input_size, 256)
        self.norm_cnn = nn.LayerNorm(256)
        # Input for the scalar values
        self.fc2 = nn.Linear(num_scalars, 256)
        self.norm_scalar = nn.LayerNorm(256)
        # Concat output of fc1 (conv) and fc2 (scalar)
        self.fc3 = nn.Linear(256 * 2, 256)
        self.norm2 = nn.LayerNorm(256)
        # Output layer
        self.fc4 = nn.Linear(256, num_classes)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, obs):
        x_scalar = obs[0].float()
        prob_matrix = obs[1]
        # Forward pass for image input
        x = self.conv1(prob_matrix)
        x = self.conv2(x)

        x = torch.flatten(x, 0)
        # Add batch dimension
        # x = x.view(x.size(0), -1)
        x_matrix = relu(self.fc1(x))
        x_matrix = self.norm_cnn(x_matrix)
        x_scalar = relu(self.fc2(x_scalar))
        x_scalar = self.norm_scalar(x_scalar)
        x = torch.cat((x_matrix, x_scalar), dim=-1)

        # Forward pass through fully connected layers
        # x = self.dropout(relu(self.fc1(x)))
        x = relu(self.fc3(x))
        x = self.norm2(x) # TODO: Should remove or keep ??
        x = self.fc4(x)
        return self.softmax(x)