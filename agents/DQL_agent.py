import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import adam, sgd, Nadam, RMSprop
from keras.layers import Activation
from keras import optimizers
import time


# Rocket lander dimensions

#N_ACTION = 7
#STATE_DIM = 10

# Lunar lander dimensions

N_ACTION = 4
STATE_DIM = 8


class Agent(object):
    """ Abstract class for agents
    """
    def __init__(self, skip_frame, epsilon_init, epsilon_final, n_action=N_ACTION):
        self.n_action = n_action
        self.epsilon = epsilon_init
        self.epsilon_init = epsilon_init
        self.epsilon_final = epsilon_final

        self.skip_frame = skip_frame
        self.count_frame = 0
        self.repeated_action = 0
        
    def decrease_epsilon(self, game, ngames, verbose=False):
        step = 3 * (self.epsilon_init - self.epsilon_final) / ngames
        self.epsilon -= step
        self.epsilon = max(self.epsilon, self.epsilon_final)

        if verbose:
            print('**** EPSILON = ', self.epsilon)

        pass

    def act(self, s, train=True):
        """ This function should return the next action to do:
        an integer between 0 and n_action (not included) with a random exploration of epsilon
        """
        if train: 

            if self.count_frame == 0: # taking new action according to epsilon greedy policy
                if np.random.rand() <= self.epsilon:
                    a = np.random.randint(0, self.n_action, size=1)[0]
                else:
                    a = self.learned_act(s)
                self.repeated_action = a
                self.count_frame += 1

            else: # repeat past action for `skip_frame` iterations
                a = self.repeated_action
                self.count_frame += 1
                if self.count_frame == self.skip_frame:
                    self.count_frame = 0

        else: # following best policy
            a = self.learned_act(s)

        return a

    def learned_act(self,s):
        """ Act via the policy of the agent, from a given state s
        it proposes an action a
        """
        pass

    def reinforce(self, s, n_s, a, r, game_over_):
        """ This function is the core of the learning algorithm. 
        It takes as an input the current state s_, the next state n_s_
        the action a_ used to move from s_ to n_s_ and the reward r_.
        
        Its goal is to learn a policy.
        """
        pass

    def save(self):
        """ This function returns basic stats if applicable: the
        loss and/or the model
        """
        pass

    def load(self):
        """ This function allows to restore a model
        """
        pass


class Memory(object):
    """ Class to stores moves in a replay buffer
    """
    def __init__(self, max_memory=5000):
        self.max_memory = max_memory
        self.memory = list()    

    def remember(self, m):
        self.memory.append(m)
        if len(self.memory)>self.max_memory:
            self.memory = self.memory[1:]
        pass

    def random_access(self):
        return self.memory[np.random.choice(len(self.memory))]

    def get_length(self):
        return len(self.memory)


class DQN(Agent):
    """ Implentation of a DeepQ learning agent
    """
    def __init__(self, skip_frame, epsilon_init, epsilon_final, memory_size, batch_size, discount, state_dim=STATE_DIM):
        super(DQN, self).__init__(skip_frame, epsilon_init, epsilon_final)
        self.discount = discount 
        self.state_dim = state_dim 
        self.batch_size = batch_size
        self.memory = Memory(memory_size) 
        self.avgQ = 0

    def learned_act(self, s):
        predicted_Q = self.model.predict(s.reshape(1, self.state_dim))
        self.avgQ = np.mean(predicted_Q)
        return np.argmax(predicted_Q)

    def store_meanq(self):
        return self.avgQ

    def reinforce(self, s_, n_s_, a_, r_, game_over_):

        assert (len(s_) == self.state_dim) and (len(n_s_) == self.state_dim)

        # first memorize
        self.memory.remember([s_, n_s_, a_, r_, game_over_])

        # second learn from the pool
        bs = min(self.batch_size, self.memory.get_length()) 
        input_states = np.zeros((bs, self.state_dim))
        target_q = np.zeros((bs, self.n_action))

        for i in range(bs):
            s_, n_s_, a_, r_, game_over_ = self.memory.random_access()
            input_states[i] = s_

            # get artificial targets for the actions
            target_q[i] = self.model.predict(s_.reshape(1, self.state_dim))

            # updating real target for action of interest `a_`
            if game_over_:
                target_q[i][a_] = r_
            else:
                maxi = np.max(self.model.predict(n_s_.reshape(1, self.state_dim)))
                target_q[i][a_] = r_ + self.discount * maxi

        # clip target to avoid exploiding gradients
        target_q = np.clip(target_q, -200, 200) 
        l = self.model.train_on_batch(input_states, target_q)

        return l

   
class DQN_FC(DQN):
    def __init__(self, skip_frame, epsilon_init, epsilon_final, memory_size, batch_size, discount, train):
        super(DQN_FC, self).__init__(skip_frame, epsilon_init, epsilon_final, memory_size, batch_size, discount) 

        if train: # create NN
            # NN model
            model = Sequential()
            model.add(Dense(32, activation='relu', input_dim=self.state_dim))
            model.add(Dense(self.n_action, activation='linear'))
            ## optimizer
            #optim = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
            #optim = optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
            #optim = optimizers.SGD(lr=0.001, momentum=0.0, decay=0.0, nesterov=False)
            optim = optimizers.RMSprop(lr=0.00025, rho=0.9, epsilon=None, decay=0.0)
            model.compile(optimizer=optim, loss='mse')
            self.model = model

        else: # load pre-trained model
            self.load()

    def save(self, name_model='model_DQN_FC.h5'):
        path = './saved_models/'
        self.model.save(path + name_model)
        pass
            
    def load(self, name_model='model_DQN_FC.h5'):
        path = './saved_models/'
        self.model = load_model(path + name_model)
        pass


class DoubleDQN(Agent):
    def __init__(self, skip_frame, epsilon_init, epsilon_final, memory_size, batch_size, discount, state_dim=STATE_DIM):
        super(DoubleDQN, self).__init__(skip_frame, epsilon_init, epsilon_final)
        self.discount = discount 
        self.state_dim = state_dim 
        self.batch_size = batch_size
        self.memory = Memory(memory_size) 
        self.avgQ = 0

    def learned_act(self, s):
        predicted_Q1 = self.model1.predict(s.reshape(1, self.state_dim))
        predicted_Q2 = self.model2.predict(s.reshape(1, self.state_dim))
        # using mean of both networks to act
        predicted_Q = 1./2 * (predicted_Q1 + predicted_Q2)
        self.avgQ = np.mean(predicted_Q)
        return np.argmax(predicted_Q)

    def store_meanq(self):
        return self.avgQ

    def reinforce(self, s_, n_s_, a_, r_, game_over_):

        assert (len(s_) == self.state_dim) and (len(n_s_) == self.state_dim)

        # first memorize the states
        self.memory.remember([s_, n_s_, a_, r_, game_over_])

        # second learn from the pool
        wesh = min(self.batch_size, self.memory.get_length()) 
        input_states = np.zeros((wesh, self.state_dim))
        target_q = np.zeros((wesh, self.n_action))

        # updatting network 1 with evaluation from network 2
        if np.random.binomial(1, 0.5) < 0.5:

            for i in range(wesh):
                s_, n_s_, a_, r_, game_over_ = self.memory.random_access()
                input_states[i] = s_

                # artificial targets for the other actions
                target_q[i] = self.model1.predict(s_.reshape(1, self.state_dim))

                # updating real target for action `a_`
                if game_over_:
                    target_q[i][a_] = r_
                else:
                    best_action = np.argmax(self.model1.predict(n_s_.reshape(1, self.state_dim)))
                    evaluation = self.model2.predict(n_s_.reshape(1, self.state_dim))[0, best_action]
                    target_q[i][a_] = r_ + self.discount * evaluation

            target_q = np.clip(target_q, -200, 200) # clip target to avoid exploiding gradients
            l = self.model1.train_on_batch(input_states, target_q)

        # updatting network 2 with evaluation from network 1
        else:

            for i in range(wesh):
                s_, n_s_, a_, r_, game_over_ = self.memory.random_access()
                input_states[i] = s_

                # artificial targets for the other actions
                target_q[i] = self.model2.predict(s_.reshape(1, self.state_dim))

                # updating real target for action `a_`
                if game_over_:
                    target_q[i][a_] = r_
                else:
                    best_action = np.argmax(self.model2.predict(n_s_.reshape(1, self.state_dim)))
                    evaluation = self.model1.predict(n_s_.reshape(1, self.state_dim))[0, best_action]
                    target_q[i][a_] = r_ + self.discount * evaluation

            target_q = np.clip(target_q, -200, 200) # clip target to avoid exploiding gradients
            l = self.model2.train_on_batch(input_states, target_q)

        return l


class DoubleDQN_FC(DoubleDQN):

    def __init__(self, skip_frame, epsilon_init, epsilon_final, memory_size, batch_size, discount, train, my_model):
        super(DoubleDQN_FC, self).__init__(skip_frame, epsilon_init, epsilon_final, memory_size, batch_size, discount) 

        if train: # creates NN
            # first NN model
            model1 = Sequential()
            model1.add(Dense(512, activation='relu', input_dim=self.state_dim))
            model1.add(Dense(256, activation='relu', input_dim=self.state_dim))
            model1.add(Dense(self.n_action, activation='linear'))
            # second NN model
            model2 = Sequential()
            model2.add(Dense(512, activation='relu', input_dim=self.state_dim))
            model2.add(Dense(256, activation='relu', input_dim=self.state_dim))
            model2.add(Dense(self.n_action, activation='linear'))
            ## optimizer
            optim = optimizers.Adam(lr=0.003, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
            #optim = optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
            #optim = optimizers.SGD(lr=0.001, momentum=0.0, decay=0.0, nesterov=False)
            #optim = optimizers.RMSprop(lr=0.00025, rho=0.9, epsilon=None, decay=0.0)
            model1.compile(optimizer=optim, loss='mse')
            model2.compile(optimizer=optim, loss='mse')
            self.model1 = model1
            self.model2 = model2
        else: # load pre-trained model
            self.load(my_model)

    def save(self, name_model1='model_DoubleDQN_FC_1.h5', name_model2='model_DoubleDQN_FC_2.h5'):
        path = './saved_models/'
        self.model1.save(path + name_model1)
        self.model2.save(path + name_model2)
        pass
            
    def load(self, model):
        path = './saved_models/'
        name_model1 = model + '_1.h5'
        name_model2 = model + '_2.h5'
        self.model1 = load_model(path + name_model1)
        self.model2 = load_model(path + name_model2)
        pass