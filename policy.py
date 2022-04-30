import torch.nn as nn
import torch.nn.functional as function


class Policy(nn.Module):
    def __init__(self, state_size = 4, action_size=2):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_size, 16)
        self.fc2 = nn.Linear(16, action_size)

    def forward(self, x):
        x = function.relu(self.fc1(x))
        x = self.fc2(x)
        return function.softmax(x, dim=1)
