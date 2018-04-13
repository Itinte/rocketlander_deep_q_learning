import argparse
import sys
import gym
from gym import wrappers, logger
import runner
import numpy as np
import matplotlib.pylab as plt

import io
import base64
import json

from keras.models import Sequential
from keras.layers import Dense
#from keras.models import Sequential #model_from_json
from keras.layers.core import Dense
from keras.optimizers import sgd

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


parser = argparse.ArgumentParser(description='RL running machine')
parser.add_argument('--env', metavar='ENV_CLASS', default='RocketLander', type=str, help='Class to use for the environment.')

parser.add_argument('--train', action='store_true', help='Verbose mode')
parser.add_argument('--load_model', type=str, help='Verbose mode')

parser.add_argument('--niter', type=int, metavar='n', default='3000', help='number of iterations to simulate')
parser.add_argument('--ngames', type=int, metavar='n', default='5', help='number of iterations to simulate')

parser.add_argument('--verbose', action='store_true', help='Display cumulative results at each step')
parser.add_argument('--display', action='store_true', help='Show 2d animation')


def main():

    np.random.seed(1)

    args = parser.parse_args()
    if args.train:
        train = True
    else:
        train = False 
    
    print("Running a single instance simulation...")
    agent_class = DoubleDQN_FC(skip_frame=4,
                         epsilon_init=1,
                         epsilon_final=0.25,
                         memory_size=2**14, #16384
                         batch_size=64,
                         discount=0.99, 
                         train=train, 
                         my_model=args.load_model)

    print([(key, agent_class.__dict__[key]) for key in agent_class.__dict__.keys()])
    my_runner = runner.Runner(args.env, agent_class, args.verbose)
    if not args.display:
        final_reward, loss_evo, reward_evo, avgq_evo = my_runner.loop(args.ngames, args.niter, train)
    else:
        final_reward, loss_evo, reward_evo, avgq_evo = my_runner.loop(args.ngames, args.niter,
            train, render=True)
    print("Obtained a final reward of {}".format(final_reward))


    ##
    save_path = './figs/'

    if train:
        # saving figures
        plt.plot(loss_evo)
        plt.title('loss evolution')
        #plt.show()
        plt.savefig(save_path + 'loss_evolution.png')
        plt.clf()

        plt.plot(avgq_evo)
        plt.title('average Q evolution')
        #plt.show()
        plt.savefig(save_path + 'mean_q_evolution.png')
        plt.clf()

        plt.plot(np.arange(args.ngames),reward_evo)
        plt.title('reward evolution')
        #plt.show()
        plt.savefig(save_path + 'reward_evolution.png')
        plt.clf()


if __name__ == "__main__":
    main()