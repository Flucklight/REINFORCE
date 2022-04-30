import torch
import torch.optim as optim
from torch.distributions import Categorical
from policy import Policy


class Agent:
    def __init__(self, state_size, action_size, learning_rate, device):
        self.action_size = action_size
        self.state_size = state_size
        self.device = device

        # Q-Network
        self.policy_net = Policy(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        self.saved_log_probs = []
        self.rewards = []

    def step(self, reward, log_probs):
        # Save rewards and log_probs
        self.rewards.append(reward)
        self.saved_log_probs.append(log_probs)

    def act(self, state):
        """Returns actions for given state as per current policy."""

        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        probs = self.policy_net(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)

    def reset(self):
        self.saved_log_probs = []
        self.rewards = []

    def learn(self, gamma):
        discounts = [gamma ** i for i in range(len(self.rewards) + 1)]
        R = sum([a * b for a, b in zip(discounts, self.rewards)])

        policy_loss = []
        for log_prob in self.saved_log_probs:
            policy_loss.append(-log_prob * R)
        policy_loss = torch.cat(policy_loss).sum()

        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
