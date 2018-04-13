import argparse
import sys
import gym
from gym import wrappers, logger
import runner
import numpy as np

## Gym environements
from gym.envs.box2d.rocket_lander import RocketLander
from gym.envs.box2d.lunar_lander import LunarLander

## Agents
from agents.random_agent import RandomAgent
from agents.observation_agent import ObsAgent
from agents.actor_critic import ACAgent
from agents.epsi_greedy import EGAgent
from agents.policy_agent import PolicyAgent
from agents.tdq_agent import TDQAgent
from agents.sarsa_agent import SarsaAgent
from agents.DQL_agent import Agent, Memory
from agents.DQL_agent import DQN, DQN_FC 
from agents.DQL_agent import DoubleDQN, DoubleDQN_FC 

## This is the machinnery that runs your agent in an environment.

class Runner:
	def __init__(self, environment_name, agent, verbose=False):
		self.env = eval(environment_name)()
		self.agent = agent
		self.verbose = verbose
		self.loss_evo = []
		self.reward_evo = []
		self.avgq_evo = []

	def loop(self, ngames, niter, train, render=False):

		"""
		# saving network architecture
		save_path = './save/deep_q_learning/'
		with open(save_path + 'NN_summary.txt','w') as fh:
			# Pass the file handle in as a lambda function to make it callable
			self.agent.model.summary(print_fn=lambda x: fh.write(x + '\n'))
		"""

		cumul_reward = 0

		for g in range(1, ngames +1):
			i = 0
			observation = self.env.reset()
			game_reward = 0
			done = False

			while not done and i < niter:

				if render: # display animation
					self.env.render() 

				i += 1
				action = self.agent.act(observation, train=train)
				avgq = self.agent.store_meanq()
				prev_observation = observation
				observation, reward, done, info = self.env.step(action)

				if train:
					loss = self.agent.reinforce(prev_observation, observation,  action, reward, done)
					self.loss_evo.append(loss)

				#if (avgq != 0) and (g > ngames/2): #dont store the avg Q when the action is random 
				self.avgq_evo.append(avgq)
				game_reward += reward

				if self.verbose:
					print("Game {}, Simulation step {}:".format(g, i))
					print(" -> observation: {}".format(np.round(observation, 2)))
					print(" ->      action: {}".format(action))
					print(" ->      reward: {}".format(reward))
					print(" ->  cum_reward: {}".format(cumul_reward))
					if train:
						print(" ->        loss: {}".format(loss))

			if self.verbose:
				print(" ### Finished game number: {}, game reward: {} ###".format(g, game_reward))
				print("")
			self.reward_evo.append(game_reward)
			cumul_reward += game_reward

			# decrease epsilon
			self.agent.decrease_epsilon(g, ngames) 


		# saving model when training
		if train:
			self.agent.save()

		return cumul_reward, self.loss_evo, self.reward_evo, self.avgq_evo