import torch
import torch.nn as nn
from torch.nn.functional import relu


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
            nn.ReLU(),
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
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        # Fully connected layers
        self.fc1_input_size = (
            32 * (matrix_shape[0] // 4) * (matrix_shape[1] // 4) + num_scalars
        )
        self.fc1 = nn.Linear(self.fc1_input_size, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)

        # Dropout
        self.dropout = nn.Dropout(0.5)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, obs):
        x_scalar = obs[0]
        prob_matrix = torch.from_numpy(obs[1]).unsqueeze(0).float().to(self.device)
        # Forward pass for image input
        x = self.conv1(prob_matrix)
        x = self.conv2(x)

        x = torch.flatten(x, 0)
        # Add batch dimension
        # x = x.view(x.size(0), -1)
        x = torch.cat((x, x_scalar), dim=-1)

        # Forward pass through fully connected layers
        x = self.dropout(relu(self.fc1(x)))
        x = relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)