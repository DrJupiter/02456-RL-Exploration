import torch
from action_embed import obs_act_embed
from ppo import sample_action

def save_model(model, PATH):
    """
    Saves a model, needs: (model, PATH)
    """
    torch.save(model.state_dict(), PATH)
    print("model saved at:", PATH)

def load_model(TheModelClass, PATH,  *args, **kwargs):
    """
    Loads a new model, needs: (TheModelClass, PATH, *args, **kwargs)\\
    returns model
    """
    model = TheModelClass(*args, **kwargs)
    model.load_state_dict(torch.load(PATH))
    model.eval()

    return model

def storage_tensor(len_trajec, N_act, dim):
    """
    Stores different lists. returns the different models as so: \\
    list_log_prop, list_reward_prime, list_value_fnc, list_acts, list_obs, list_act_bonus, list_ns_bonus
    """
    # original log props
    with torch.no_grad():
        list_log_prop =     torch.zeros((len_trajec, N_act)).cuda().double()

        # rewards
        list_reward_prime = torch.zeros((len_trajec)).cuda().double()
        list_value_fnc =    torch.zeros(len_trajec + 1).cuda().double()

        # state action lists
        list_acts =         torch.zeros(len_trajec).cuda().long()
        list_obs =          torch.zeros((len_trajec + 1, dim[2],dim[1],dim[0])).cuda().double() # 1 extra dim for the obs loop

        # RND values
        list_act_bonus =    torch.zeros((len_trajec, N_act)).cuda().double()
        list_act_norm =    torch.zeros((len_trajec, N_act)).cuda().double()
        list_ns_bonus =     torch.zeros((len_trajec)).cuda().double()

    return list_log_prop, list_reward_prime, list_value_fnc, list_acts, list_obs, list_act_bonus, list_ns_bonus, list_act_norm

def storage_list(len_trajec):
    """
    Stores different lists. returns the different models as so: \\
    list_log_prop, list_reward_prime, list_value_fnc, list_acts, list_obs, list_act_bonus, list_ns_bonus
    """
    # original log props
    
    list_log_prop     = [None] * len_trajec

    # rewards
    list_reward_prime = [None] * len_trajec
    list_value_fnc    = [None] * (len_trajec + 1)
    list_masks     = [None] * len_trajec 

    # state action lists
    list_acts         = [None] * len_trajec        
    list_obs          = [None] * (len_trajec + 1) 

    # RND values
    list_act_bonus    = [None] * len_trajec 
    list_act_norm     = [None] * len_trajec 
    list_ns_bonus     = [None] * len_trajec 

    return list_log_prop, list_reward_prime, list_value_fnc, list_acts, list_obs, list_act_bonus, list_ns_bonus, list_act_norm, list_masks


def compute_gae(values, rewards, masks, len_trajec, gamma=0.99, tau=0.95):
    """
    GAE = ADVANTAGE \\
    V(s...s_T-1) -> values \\
    V(s_T) -> next_value \\
    r'...r'_T -> rewards \\
    masks -> done

    BUG?: Adv = gae - vals
    """
    
    # masks

    gae = 0
    returns = [None] * len_trajec
    # returns = torch.zeros((len_trajec)).cuda().double()
    with torch.no_grad(): 
        for step in reversed(range(len_trajec)):
            delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
            gae = delta + gamma * tau * masks[step] * gae
            returns[step] = gae + values[step]
        
    return returns


### Visualize

def single_optimal_move(observation, action_size, dim, PPO_model, RND_ACT_model, env): # Remove stuff from this to it matches description
    """
    Plays optimal action\\
    doesnt actually vizuale, is just named like this cause of the place it is used\\
    takes: obs\\
    returns: obs
    """

    ### Policy
    policy_out, v_t = PPO_model(observation.detach().clone()) # ADD seperate CNN ? maybe
    
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
    pred_act, target_act = RND_ACT_model(obs_action_embeddings.cuda().double())
    
    act_bonus = ((pred_act-target_act)**2).sum(axis = 0)/action_size # we divide by action_size, because the expectation would be magnified by the output dim by a factor of the output dim.
    # print("action bonus size",act_bonus.size(),"expected size 18")

    # normalize action bonus, we are not 100% sure if this is the correct form of normalization
    act_norm = (act_bonus - act_bonus.mean())/act_bonus.std() # BUG?: clone and detach act_bonus for loss later
    
    # get action from sample of beta
    action = torch.argmax(torch.nn.functional.softmax(policy_out + act_norm, dim = 1)) #BUG?: no detach needed here (i think) as we dont run backward on the action in any way
    
    # perform action
    observation, env_reward, done, _ = env.step(action)

    # For frameskip and consistency
    env.step(0)
    
    ## NSB 
    # Get RND prediction and target nets for reward
    observation = torch.tensor(observation).unsqueeze(0).cuda().double()
    observation = observation.permute((0,3,2,1)) 

    if done:    
        observation = env.reset()
        observation = torch.tensor(observation).unsqueeze(0).cuda().double()
        observation = observation.permute((0,3,2,1)) 
    # Return stuff, so we can save it
    return observation 

def visualize_play(action_size, dim, PPO_model, RND_ACT_model, env, observation = None, N_moves = 1000):
    """
    Plays optimal action, and vizualizes it\\
    takes : observation, reset, N_moves\\
    reset = should it start from the beginning? (If reset = False, then pass in obs, else it doesnt matter)\\
    N_moves = number of moves before end of visuzalization
    """
    if observation is None:
        observation = torch.tensor(env.reset()).permute((2,1,0)).unsqueeze(0).cuda().double()  
          
    else:
        assert observation.size() == (1,3,160,210), "wrong obs input"

    for i in range(N_moves):
        env.render()
        observation = single_optimal_move(observation, action_size, dim, PPO_model, RND_ACT_model, env)

    print("Done visualizing")


def stack_lists(*lists):

    stacked_lists = []
    
    for lis in lists:
        
        stacked_lists.append(torch.stack(lis))

    return stacked_lists
