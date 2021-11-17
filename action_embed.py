import numpy as np
import torch
from torch import tensor

def act_embed(action_num: int, in_dims: tuple, N_acts: int):

    in_dims = in_dims[:2]
    act_embed = torch.zeros(in_dims)
    
    m = in_dims[0]
    k = N_acts
    rate = m//k

    act_embed[int(action_num*rate),:] = 1
    return act_embed # same dims as in_dims[:2], always same sum

def obs_act_embed(action_num: list, in_dims: tuple, N_acts: int, obs: tensor):

    two_d_act_embed = (act_embed(action_num, in_dims, N_acts)).unsqueeze(2).unsqueeze(0)
    
    print(obs.size())
    print(two_d_act_embed.size())
    obs_act_embed = torch.cat((obs,two_d_act_embed),3)
    
    return obs_act_embed

if __name__ == "__main__":
    print(obs_act_embed(3, (210,160,3), 18, torch.zeros(210,160,3)).shape)
    print("expected size: (210,160,4)")