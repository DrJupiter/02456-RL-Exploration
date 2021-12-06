from typing import Tuple
from torch.nn import init
import numpy as np
from torch import nn

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
"""
In RND there are 2 networks:
- Target Network: generates a constant output for a given state
- Prediction network: tries to predict the target network's output
""" 


from math import floor

def cnn_dims(data_size, kernel_size, stride, padding, filters):
    """
    -> out 

    feature out can be calculated by product(out[4:7])
    """
    _batch, in_channel, height, width = data_size
    out_height = floor((height + 2 * padding[1] - (kernel_size[1]-1)-1)
                             /stride[1] + 1)
    out_width= floor((width + 2 * padding[0] - (kernel_size[0]-1)-1)
                             /stride[0] + 1)
    return [in_channel,kernel_size,stride,padding,filters, out_height,out_width]



#18 (210, 160, 3)

class RNDModel(nn.Module):
    def __init__(self, input_size: Tuple, output_size: int):
        super(RNDModel, self).__init__()

        self.input_size = input_size # WxHxC reshaped into CxHxW. Default 210x160x3
        self.output_size = output_size 

        feature_output = 576 #2**11 * 11 # the symmetry # still very cursed

        # Prediction network
        self.predictor = nn.Sequential(
            nn.Conv2d(
                in_channels=input_size[2], # something
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
            nn.Linear(512, output_size)
        )

       
        # Target network (Never updated)
        self.target = nn.Sequential(
            nn.Conv2d(
                in_channels=input_size[2],
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
            nn.Linear(512, output_size)
        )

        # Initialize the weights and biases
        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

        # Set that target network is not trainable
        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, next_obs):
        target_feature = self.target(next_obs)
        predict_feature = self.predictor(next_obs)

        return predict_feature, target_feature

# https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/master/RND%20Montezuma's%20revenge%20PyTorch/model.py
