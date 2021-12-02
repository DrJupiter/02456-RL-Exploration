
import torch
from torch import tensor
from torch import optim
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
print(f"the action size is {action_size} and the input dimensionality is {dim}")

# import from other files
from ppo import PolicyNet, sample_action
from rnd import RNDModel
from action_embed import obs_act_embed # act_embed

from utils import compute_gae, storage_list, save_model, load_model, visualize_play 

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


## lists
LEN_TRAJECTORY = 10
BATCH_SIZE_RND = 4

# Input order
# observation, action.unsqueeze(), policy_out.log(), reward, v_t, act_norm, next_state_bonus



def single_loss_ppo(ratio, reward, A, V, epsilon = 0.2):
    """
    l_p = min(r_t * A_t,clip(r_t,1-ε,1+ε)A_t)
    """

    l_clip = torch.min(ratio * A, torch.clamp(ratio, 1-epsilon, 1+epsilon) * A)
    l_critic = (reward - V)**2
    l_p = 0.5 * l_critic + l_clip # BUG: add entropy
    return l_p 

def update_rnd(actb,actions,nsb):
    RND_ACT_Optim.zero_grad()
    RND_NSB_Optim.zero_grad()
    
    #sequential_index = torch.arange(0, actb.size(0))
    sequential_index = torch.arange(0, len(actb))
    loss_act = torch.zeros(len(actb)).cuda()
    for i, a in zip(sequential_index,actions):
        loss_act[i] = actb[i][a]
    loss_act = loss_act.mean()

    loss_nsb = sum(nsb)/len(nsb)

#    loss_act = actb[sequential_index,actions].mean()
#    loss_nsb = nsb.mean()
    loss_act.backward()
    loss_nsb.backward()
    
    RND_ACT_Optim.step()
    RND_NSB_Optim.step()

def update_ppo(states, actions, log_probs, rewards, values, actnorm):
    """
    1. Take some values which should all be with no grad.
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
    advantages = compute_gae(values, rewards, torch.ones(len(rewards)),LEN_TRAJECTORY) #BUG: it makes no sense to use these sizes. What should the final GAE value be defined by?

    # TODO: perform a permutation
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

    batchsize = tensor(4)
    assert batchsize <= LEN_TRAJECTORY, "Batch size shouldn't be larger, than the trajectory length"

    batchtes = int(torch.ceil(LEN_TRAJECTORY / batchsize))
    # TODO: repeat this step equal to some number we pass in (makes sense as we only take small steps)
    for i in range(batchtes):
    
        # find indexs for batches
        index = slice(i * batchsize, min((i+1) * batchsize, LEN_TRAJECTORY))

        ### FIND RATIO
        # Get the parameters for the distribution
        parameters, values_now = PPO(states[index])
        # Find the log_prop for the current model
        # we apply log directly, since we use a multinomial distribution to sample actions
        sequential_index = torch.arange(0, parameters.size(0))

        # find chosen actions
        chosen_actions = actions[index]

        log_probs_now = (parameters[sequential_index,chosen_actions] + actnorm[sequential_index, actions]).log()

        # Find ratio pi(a|s)_now/pi(a|s)_old
        ratios = (log_probs_now - log_probs[sequential_index,chosen_actions]).exp()     

        ### Get PPO loss + actb + nsb
        loss = single_loss_ppo(ratios, rewards[index], advantages[index],values_now).mean()
        # BUG?: pass in all values, except the final one for which we don't know the reward, but we use to calculate the gae.

        # zero grad all optimizers
        PPO_Optim.zero_grad()

        # backwards
        loss.backward()
        
        # step with all optimizers
        PPO_Optim.step()

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
    # No grad => no need to detach and clone
    policy_out, v_t = PPO(observation)

    ## Find obs_action embeddings
    # Grad for update actrnd during runtime => detach & clone
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
    with torch.no_grad():
        act_norm = (act_bonus - act_bonus.mean())/act_bonus.std() # BUG?: clone and detach act_bonus for loss later
    
    # get action from sample of beta
        beta_param = F.softmax(policy_out + act_norm)
        action = sample_action(beta_param) #BUG?: no detach needed here (i think) as we dont run backward on the action in any way
    
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
    with torch.no_grad():
        reward = next_state_bonus.detach().clone() + env_reward # BUG?: Deatch the next state bonus here to avoid double update and possibly nonsensical gradients

    ### reset if model died or achieved goal state
    if done:    
        observation = env.reset()
        observation = tensor(observation).unsqueeze(0).cuda().double()
        observation = observation.permute((0,3,2,1)) 

    # Return stuff, so we can save it
    return observation, action.unsqueeze(0), beta_param.log().squeeze(0), reward, v_t.squeeze(0), act_bonus, next_state_bonus, act_norm
    

def play(EPOCHS, observation):
    # observation = env.reset()
    
    for _ in range(EPOCHS):

        list_log_prop, list_reward_prime, list_value_fnc, list_acts, list_obs,list_act_bonus,list_ns_bonus, list_actnorm = storage_list(LEN_TRAJECTORY)

        
    

        # setting initial obs
        observation = tensor(env.reset()).cuda().double().permute((2,1,0)).unsqueeze(0)
        
        list_obs[0] = observation
        #print(list_obs[0].size())
        
        ## Loop through the trajectory using prior ops
        PPO.eval()
        for t in range(LEN_TRAJECTORY):

            list_obs[t+1], list_acts[t], list_log_prop[t], list_reward_prime[t], list_value_fnc[t], list_act_bonus[t], list_ns_bonus[t], list_actnorm[t] = single_pass(list_obs[t])
            ## UPDATE RND
            # in order for observation to be updated
            if t != 0 and t % BATCH_SIZE_RND == 0:
                print(t-BATCH_SIZE_RND,t)
                index = slice(t-BATCH_SIZE_RND,t)
                #x,y,z = torch.stack(list_act_bonus[index]),torch.stack(list_acts[index]),torch.stack(list_ns_bonus[index])
                x,y,z = list_act_bonus[index],list_acts[index],list_ns_bonus[index]
                     
                     
                update_rnd(x,y,z)

        ## Final stuff after loop
        # Update RND's for the missing steps (In the case where LEN_TRAJECTORY isnt divisible by BATCH_SIZE_RND)
        index = slice(t - (t % BATCH_SIZE_RND),t)
        update_rnd(list_act_bonus[index],list_acts[index],list_ns_bonus[index])

        # Add final value
        list_value_fnc[LEN_TRAJECTORY] = PPO.forward_critic(list_obs[LEN_TRAJECTORY].unsqueeze(0))



        ## Update nets
        PPO.train()
        update_ppo(list_obs, list_acts, list_log_prop, list_reward_prime, list_value_fnc, list_act_bonus, list_ns_bonus, list_actnorm)

        #observation = list_obs[-1].detach().clone()

    
#save_model(PPO, PATH = "./models/PPO_test_1")
model = load_model(PolicyNet,PATH = "./models/PPO_test_1", n_inputs = dim, n_outputs = action_size, )
model.cuda().double()


#print(model(observation.permute((2,1,0)).unsqueeze(0)))
play(5, observation.permute((2,1,0)).unsqueeze(0))

visualize_play(action_size, dim, PPO, RND_ACT, env, observation = None, reset = True, N_moves = 1000)


env.close()
