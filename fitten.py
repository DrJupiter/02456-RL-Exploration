
import torch
from torch import tensor

import gym

env_name = "MontezumaRevenge-v0"
env = gym.make(env_name)

action_size = env.action_space.n
observation = env.reset()

dim = observation.shape

print(action_size,dim)

from rnd import RNDModel
from action_embed import act_embed #obs_act_embed

RND_NS = RNDModel(dim,action_size) # for rewards
# +1, becasue we can make it (3+1)xHxW via action embedding
RND_ACT = RNDModel((dim[0],dim[1],dim[2]+1),action_size) # for action bonus
RND_NS.cuda()


for _ in range(1):
#    env.render()
    action = env.action_space.sample() # agent act
    observation, reward, done, _ = env.step(action)
    observation = tensor(observation).permute((2,1,0)).unsqueeze(0)
    observation = observation.cuda().float()
    observation_clone = observation.detach().clone()

    action_embedding = act_embed(1,dim, action_size) 
    
    print(observation.size())
    pred, target = RND_NS(observation)
    print(((pred-target)**2).sum())
     
    if done:    
        observation = env.reset()

env.close()
