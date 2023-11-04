import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class NN1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))


class NN2(nn.Module):
    def __init__(self, input_size, hidden_layers, num_classes):
        super(NN2, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_layers)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_layers, num_classes)

    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.relu(x1)
        out = self.fc2(x2)
        return out


class NN3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 128, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(128, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x