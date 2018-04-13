import numpy as np

class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation):
        return self.action_space.sample()

    def reward(self, observation, action, reward):
        # where the agent is learning
        pass

    def save_model(self, path):
    	# saving model
    	pass

    def load_model(self, path):
    	# loading previously saved model
    	pass