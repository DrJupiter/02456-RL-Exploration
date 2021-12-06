import torch
from torch import tensor
from torch import nn
import torch.nn.functional as F
from torch import optim

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

# act_probs from softmax on output from policy net
# as in the paper
def sample_action(act_probs: tensor, num_samples = 1) -> int:
    return torch.multinomial(act_probs,num_samples=num_samples).squeeze()

# Baseline policy net from course lecture 8
class PolicyNet(nn.Module): #PO
    """Policy network"""

    def __init__(self, n_inputs, n_outputs):
        super(PolicyNet, self).__init__()
        # network
        self.input_size = n_inputs # WxHxC reshaped into CxHxW. Default 56x56x3
        self.output_size = n_outputs 
 
        feature_output = 576 # 2**11 * 11 # the symmetry # still very cursed
 
        # Prediction network
        self.actor = nn.Sequential(
             nn.Conv2d(
                 in_channels=self.input_size[2], # something
                 out_channels=32,
                 kernel_size=8,
                 stride=4),
             nn.ELU(),
             nn.Conv2d(
                 in_channels=32,
                 out_channels=64,
                 kernel_size=4,
                 stride=2),
             nn.ELU(),
             nn.Conv2d(
                 in_channels=64,
                 out_channels=64,
                 kernel_size=3,
                 stride=1),
             nn.ELU(),
             Flatten(),
             nn.Linear(feature_output, 512),
             nn.ELU(),
             nn.Linear(512, 512),
             nn.ELU(),
             nn.Linear(512, self.output_size)
        )

        self.critic = nn.Sequential(
             nn.Conv2d(
                 in_channels=self.input_size[2], # something
                 out_channels=32,
                 kernel_size=8,
                 stride=4),
             nn.ELU(),
             nn.Conv2d(
                 in_channels=32,
                 out_channels=64,
                 kernel_size=4,
                 stride=2),
             nn.ELU(),
             nn.Conv2d(
                 in_channels=64,
                 out_channels=64,
                 kernel_size=3,
                 stride=1),
             nn.ELU(),
             Flatten(),
             nn.Linear(feature_output, 512),
             nn.ELU(),
             nn.Linear(512, 512),
             nn.ELU(),
             nn.Linear(512, 1)
        )

        # training

    def forward(self, x):
        x_copy = x.detach().clone()
        x = self.actor(x)
        v = self.critic(x_copy)
        return F.softmax(x, dim=1), v
    
    def forward_critic(self,x):
        return self.critic(x)

    # I think this loss requires some modifications # indeed
    def loss(self, action_probabilities, returns):
        return -torch.mean(torch.mul(torch.log(action_probabilities), returns))

# https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/master/RND%20Montezuma's%20revenge%20PyTorch/model.py
