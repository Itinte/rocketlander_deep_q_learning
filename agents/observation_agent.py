import numpy as np




# AGENT THAT SIMPLY DISPLAYS THE VARIABLES OF THE GAME

class ObsAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space
        print('action space', action_space)
        print('action space shape', action_space.shape)
        print('#########################################')

    def act(self, observation):
        print('obs_act',observation)
        print('obs_act shape', observation.shape)
        return self.action_space.sample()

    def reward(self, observation, action, reward):
        # where the agent is learning
        print('obs_rew',observation)
        print('obs_rew shape', observation.shape)
        print('action',action)
        print('action shape', observation.shape)
        print('reward',action)
        pass

    def save_model(self, path):
    	# saving model
    	pass

    def load_model(self, path):
    	# loading previously saved model
    	pass
