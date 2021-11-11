import gym

env = gym.make('MontezumaRevenge-v0')
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action

    if done:
        observation = env.reset()
env.close()
