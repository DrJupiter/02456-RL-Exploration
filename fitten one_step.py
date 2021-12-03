
import torch
from torch import tensor
import torch.nn.functional as F 
from torch.optim import Adam
torch.autograd.set_detect_anomaly(True)
import gym

# make env
env_name = "MontezumaRevenge-v0"
env = gym.make(env_name)

# get ini state / obs
action_size = env.action_space.n
observation = tensor(env.reset()).cuda().double() # BUG?: take env.reset out

dim = observation.size()

# print state dims
print(action_size,dim)

# import from other files
from ppo import PolicyNet, sample_action
from rnd import RNDModel
from action_embed import obs_act_embed # act_embed


# Ini global variables
GAMMA = 0.99

# Ini nets
PPO = PolicyNet(n_inputs = dim, n_outputs = action_size)
PPO.cuda()
PPO.double()
PPO_Optim = Adam(PPO.parameters(), lr=1e-4)


RND_NSB = RNDModel(dim,action_size) # for rewards
RND_NSB.cuda()
RND_NSB.double()
RND_NSB_Optim = Adam(RND_NSB.parameters(), lr=1e-4)

# +1, becasue we can make it (3+1)xHxW via action embedding
RND_ACT = RNDModel((dim[0],dim[1],dim[2]+1),action_size) # for action bonus
RND_ACT.cuda()
RND_ACT.double()
RND_ACT_Optim = Adam(RND_ACT.parameters(), lr=1e-4)


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
LEN_TRAJECTORY = 4

# Input order
# observation, action.unsqueeze(), policy_out.log(), reward, v_t, act_norm, next_state_bonus

def storage(len_tratjec = LEN_TRAJECTORY, N_act = action_size):
    # original log props
    list_log_prop =     torch.zeros((len_tratjec, N_act)).cuda().double()

    # rewards
    list_reward_prime = torch.zeros((len_tratjec)).cuda().double()
    list_value_fnc =    torch.zeros(len_tratjec + 1).cuda().double()

    # state action lists
    list_acts =         torch.zeros(len_tratjec).cuda().long()
    list_obs =          torch.zeros((len_tratjec + 1, dim[2],dim[1],dim[0])).cuda().double() # 1 extra dim for the obs loop

    # RND values
    list_act_bonus =    torch.zeros((len_tratjec, N_act)).cuda().double()
    list_ns_bonus =     torch.zeros((len_tratjec)).cuda().double()

    return list_log_prop, list_reward_prime, list_value_fnc, list_acts, list_obs, list_act_bonus, list_ns_bonus

def compute_gae(values, rewards, masks, gamma=0.99, tau=0.95):
    """
    GAE = ADVANTAGE \\
    V(s...s_T-1) -> values \\
    V(s_T) -> next_value \\
    r'...r'_T -> rewards \\
    masks -> done

    BUG?: Adv = gae - vals
    """
    gae = 0
    returns = torch.zeros((LEN_TRAJECTORY)).cuda().double()
    
    for step in reversed(range(LEN_TRAJECTORY)):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns[step] = gae + values[step]
    
    return returns

def single_loss(ratio, reward, A, V, actb, nsb, epsilon = 0.2):
    """
    l_p = min(r_t * A_t,clip(r_t,1-ε,1+ε)A_t)
    """

    l_clip = torch.min(ratio * A, torch.clamp(ratio, 1-epsilon, 1+epsilon) * A)
    l_critic = (reward - V)**2
    l_p = 0.5 * l_critic + l_clip # BUG: add entrohy
    print("sizes")
    print(l_p , actb , nsb)
    return l_p + actb + nsb

def update(states, actions, log_probs, rewards, values, actb, nsb):
    """
    1. Take some values (states, actions and loss)
    2. compute gradients
    3. Update weights (based on loss and gradients)
    4. ...
    5. profit

    This is done by randomly selecting a state/action pair \\
    computing gradients by passing state into policy and (action bonus??) \\
    updating weights \\
    loop 
    """
    
    # Calc GAE / Advantage
    advantages = compute_gae(values.detach().clone(), rewards.detach().clone(), torch.ones(len(rewards))) #BUG: it makes no sense to use these sizes. What should the final GAE value be defined by?

    # TODO: perform a permutation
    
    batchsize = tensor(4)
    assert batchsize <= LEN_TRAJECTORY, "Batch size shouldn't be larger, than the trajectory length"

    batchtes = int(torch.ceil(LEN_TRAJECTORY / batchsize))

    # TODO: repeat this step equal to some number we pass in (makes sense as we only take small steps)
    for i in range(batchtes):
    
        # find indexs for batches
        index = slice(i * batchsize, min((i+1) * batchsize, LEN_TRAJECTORY))

        ### FIND RATIO
        # Get the parameters for the distribution
        parameters, _ = PPO(states[index])
        # Find the log_prop for the current model
        # we apply log directly, since we use a multinomial distribution to sample actions
        sequential_index = torch.arange(0, parameters.size(0))

        # find chosen actions
        chosen_actions = actions[index]

        log_probs_now = parameters.log()[sequential_index,chosen_actions]

        # Find ratio pi(a|s)_now/pi(a|s)_old
        ratios = (log_probs_now - log_probs[sequential_index,chosen_actions]).exp()     

        ### Get PPO loss + actb + nsb
        loss = single_loss(ratios, rewards[index], advantages[index], values[index], actb[index][sequential_index, chosen_actions], nsb[index]
                ).mean()                                                            #actb[index][sequential_index, chosen_actions]
        # BUG?: pass in all values, except the final one for which we don't know the reward, but we use to calculate the gae.

        # zero grad all optimizers
        PPO_Optim.zero_grad()
        RND_NSB_Optim.zero_grad()
        RND_ACT_Optim.zero_grad()

        # backwards
        loss.backward()
        
        # step with all optimizers
        PPO_Optim.step()
        RND_NSB_Optim.step()
        RND_ACT_Optim.step()

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

    ### Policy
    policy_out, v_t = PPO(observation.detach().clone()) # ADD seperate CNN ? maybe

    ## Find obs_action embeddings
    obs_action_embeddings = torch.zeros((action_size,dim[2]+1,dim[1],dim[0]))
    for a in range(action_size):
        obs_action_embeddings[a] = obs_act_embed(
            action_num = a,
            in_dims = (1, dim[1], dim[0]),
            N_acts = action_size, 
            obs = observation.detach().clone()
            ).cuda()


    ## Action bonus

    pred_act, target_act = RND_ACT(obs_action_embeddings.cuda().double())
    
    act_bonus = ((pred_act-target_act)**2).sum(axis = 0)/action_size # we divide by action_size, because the expectation would be magnified by the output dim by a factor of the output dim.
    # print("action bonus size",act_bonus.size(),"expected size 18")

    # normalize action bonus, we are not 100% sure if this is the correct form of normalization
    act_norm = (act_bonus - act_bonus.mean())/act_bonus.std() # BUG?: clone and detach act_bonus for loss later
    
    # get action from sample of beta
    action = sample_action(F.softmax(policy_out + act_norm)) #BUG?: no detach needed here (i think) as we dont run backward on the action in any way
    
    # perform action
    observation, env_reward, done, _ = env.step(action)
    
    ## NSB 
    # Get RND prediction and target nets for reward
    observation = tensor(observation).unsqueeze(0).cuda().double()
    observation = observation.permute((0,3,2,1)) 
    
    # get values
    pred_nsb, target_nsb = RND_NSB(observation)
    
    # Finds squared error between the next_state_bpnus nets
    next_state_bonus = ((pred_nsb-target_nsb)**2).sum(axis = 1) # divide by action_size for the same reason as with the act_bonus
                                                                # Sum over axis=1, cause dim is fucked
    # # loss
    # At = tensor(env_reward) + GAMMA * v_t1 - v_t0
    
    # Total reward
    reward = next_state_bonus.detach().clone() + env_reward # BUG?: Deatch the next state bonus here to avoid double update and possibly nonsensical gradients

    ### reset if model died or achieved goal state
    if done:    
        observation = env.reset()

    # Return stuff, so we can save it
    return observation, action.unsqueeze(0), policy_out.log().squeeze(0), reward, v_t.squeeze(0), act_bonus, next_state_bonus


def play(EPOCHS, observation):
    # observation = env.reset()
    
    for _ in range(EPOCHS):

        list_log_prop, list_reward_prime, list_value_fnc, list_acts, list_obs,list_act_bonus,list_ns_bonus = storage()

        # setting initial obs
        observation = tensor(env.reset()).cuda().double().permute((2,1,0)).unsqueeze(0)
        
        list_obs[0] = observation
        #print(list_obs[0].size())
        
        # Loop through the trajectory using prior ops
        PPO.eval()
        for t in range(LEN_TRAJECTORY):
            #for i in (single_pass(observation)):
            #    print(i.size())
            #exit(0)
            
            list_obs[t+1], list_acts[t], list_log_prop[t], list_reward_prime[t], list_value_fnc[t], list_act_bonus[t], list_ns_bonus[t] = single_pass(list_obs[t].unsqueeze(0))
            ## UPDATE RND
            # in order for observation to be updated
        
        # Add final value
        list_value_fnc[LEN_TRAJECTORY] = PPO.forward_critic(list_obs[LEN_TRAJECTORY].unsqueeze(0))

        # Update nets
        PPO.train()
        update(list_obs, list_acts, list_log_prop, list_reward_prime, list_value_fnc, list_act_bonus, list_ns_bonus)

        #observation = list_obs[-1].detach().clone()


play(5, observation.permute((2,1,0)).unsqueeze(0))

env.close()
