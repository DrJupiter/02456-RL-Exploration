import torch
from torch import tensor
from torch import nn
import torch.nn.functional as F
from torch import optim

# act_probs from softmax on output from policy net
# as in the paper
def sample_action(act_probs: tensor) -> int:
    return torch.multinomial(act_probs,num_samples=1).squeeze()

# Baseline policy net from course lecture 8
class PolicyNet(nn.Module):
    """Policy network"""

    def __init__(self, n_inputs, n_hidden, n_outputs, learning_rate):
        super(PolicyNet, self).__init__()
        # network
        self.hidden = nn.Linear(n_inputs, n_hidden)
        self.out = nn.Linear(n_hidden, n_outputs)
        # training
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        x = self.hidden(x)
        x = F.relu(x)
        x = self.out(x)
        return F.softmax(x, dim=1)

    # I think this loss requires some modifications # indeed
    def loss(self, action_probabilities, returns):
        return -torch.mean(torch.mul(torch.log(action_probabilities), returns))

# https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/master/RND%20Montezuma's%20revenge%20PyTorch/model.py

