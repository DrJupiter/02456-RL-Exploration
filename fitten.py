
import torch
from torch import tensor
import torch.nn.functional as F 

import gym

# make env
env_name = "MontezumaRevenge-v0"
env = gym.make(env_name)

# get ini state / obs
action_size = env.action_space.n
observation = env.reset()

dim = observation.shape

# print state dims
print(action_size,dim)

# import from other files
from ppo import PolicyNet, sample_action
from rnd import RNDModel
from action_embed import obs_act_embed # act_embed

# Ini global variables
GAMMA = 0.99

# Ini nets
PPO = PolicyNet(n_inputs = dim, n_hidden = 1, n_outputs = action_size)
RND_NSB = RNDModel(dim,action_size) # for rewards
RND_NSB.cuda()

# +1, becasue we can make it (3+1)xHxW via action embedding
RND_ACT = RNDModel((dim[0],dim[1],dim[2]+1),action_size) # for action bonus
RND_ACT.cuda()

# Kører igennem det hele 2 gange
    # GEM (run 1)
        # Run thought the trajectory decided by the policy and action bonus
            # save
                # rewards (env_reward)
                # value function (critic value function)
                # state action pairs (obs_act_embed)
                # log_prop (policy)  
                    # Så behøver vi vel ikke holde en kopi?
                # next state bonus (fra RND_NSB)
                # action bonus (fra RND_ACT)

    # Update (run 2)
        # Runs the state action pairs in random order
        # uses info from first run to calc loss
        # uses the fact that we have run it again s.t. we have the gradients
        # ...
        # profit

    # ADD TRAJECTORY 
        # sample some state
        # perform actions X times
        # use this to find discounted rewards
## lists
len_trajec = 10

# original log props
list_log_prop =     torch.zeros(len_trajec)

# rewards
list_reward_prime = torch.zeros(len_trajec)
list_value_fnc =    torch.zeros(len_trajec)

# state action lists
list_acts =         torch.zeros(len_trajec)
list_obs =          torch.zeros((len_trajec,dim[0],dim[1],dim[2]))
list_obs_tp1 =      torch.zeros((len_trajec,dim[0],dim[1],dim[2])) # we need to keep list of this?

# RND values
list_act_bonus =    torch.zeros((len_trajec, action_size))
list_ns_bonus =     torch.zeros(len_trajec)

def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
    """
    V(s...s_T-1) -> values
    V(s_T) -> next_value 
    r'...r'_T -> rewards
    masks -> done

    BUG?: Adv = gae - vals
    """
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns



def single_loss(log_prop,log_prop_now, reward, A, V, actb, nsb, epsilon = 0.2):
    """
    l_p = min(r_t * A_t,clip(r_t,1-ε,1+ε)A_t)
    """

    ratio = (log_prop_now-log_prop).exp() 
    l_clip = torch.min(ratio * A,  torch.clamp(ratio, 1-epsilon, 1+epsilon) * A).mean()
    l_critic = F.mse_loss(reward, V)
    l_p = 0.5 * l_critic + l_clip # BUG: add entrohy

    return l_p + actb + nsb

def loss(r, V, actb, nsb):

    return None

def single_pass(observation):
    """
    1. Converte observation to tensor
    2. Get π(a|s) and V(s) from PPO
    3. Embed actions and Concatenate them with the current Observation
    4. Get action bonus (r_ab^s) from ||pred_act-target_act||^2 from RND_ACT
    5. ß(a|s) = softmax(Normalize(r_ab) + π(a|s))
    6. a ∼ P(ß(a|s)): Pass ß(a|s) to the desired distribution (Multinomial) and sample an action 
    7. Perform action a
    8. return log_prop(a ~ P), observation_{t+1}, r', V(s), a, observation, action_bonus, next_state_bonus  
    """


    ### Convert observation to a tensor
    observation = tensor(observation).unsqueeze(0)

    ### Policy
    policy_out, v_t0 = PPO(observation.detach().clone().permute((0,3,2,1))) # ADD seperate CNN ? maybe

    ## Find obs_action embeddings
    obs_action_embeddings = torch.zeros((action_size,dim[2]+1,dim[1],dim[0]))
    for a in range(action_size):
        obs_action_embeddings[a] = obs_act_embed(
            action_num = a,
            in_dims = dim,
            N_acts = action_size, 
            obs = observation.detach().clone()
            ).permute((0,3,2,1)
            ).cuda()

    ## Action bonus
    pred_act, target_act = RND_ACT(obs_action_embeddings)
    act_bonus = ((pred_act-target_act)**2).sum(axis = 0)/action_size # we divide by action_size, because the expectation would be magnified by the output dim by a factor of the output dim.

    # normalize action bonus, we are not 100% sure if this is the correct form of normalization
    # BUG?: clone and detach act_bonus for loss later
    act_norm = (act_bonus - act_bonus.mean())/act_bonus.std()

    # get action from sample of beta
    action = sample_action(F.softmax(policy_out + act_norm))
    
    # perform action
    observation, env_reward, done, _ = env.step(action)
    
    ## Get RND prediction and target nets for reward
    observation = observation.permute((0,3,2,1)) 
    observation = observation.cuda().float()
    pred_nsb, target_nsb = RND_NSB(observation)

    # Finds squared error between the next_state_bpnus nets
    next_state_bonus = ((pred_nsb-target_nsb)**2).sum(axis = 0)/action_size # divide by action_size for the same reason as with the act_bonus

    v_t1 = PPO.forward_critic(observation.detach().clone())


    # # loss
    # At = tensor(env_reward) + GAMMA * v_t1 - v_t0
    
    # Total reward
    reward = next_state_bonus + env_reward



    ### reset if model died or achieved goal state
    if done:    
        observation = env.reset()




env.close()
