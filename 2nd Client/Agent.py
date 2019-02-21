import Model
import random
import numpy as np
from collections import deque
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Softmax
from keras.optimizers import Adam
import tensorflow as tf
from keras.backend import clear_session
from keras import backend as K

import os
class DQNAgent:
    def __init__(self, name, world):
        self.state_size = len(world.map.cells)*len(world.map.cells[0])
        self.action_size = 4
        self.memory = deque(
            maxlen=2000)  # double-ended queue; acts like list, but elements can be added/removed from either end
        self.gamma = 0.95  # decay or discount rate: enables agent to take into account future actions in addition to the immediate ones, but discounted at this rate
        self.epsilon = 1.0  # exploration rate: how much to act randomly; more initially than later due to epsilon decay
        self.epsilon_decay = 0.995  # decrease number of random explorations as the agent's performance (hopefully) improves over time
        self.epsilon_min = 0.01  # minimum amount of random exploration permitted
        self.learning_rate = 0.01  # rate at which NN adjusts models parameters via SGD to reduce cost
        self.model = self._build_model()  # private method
        self.name = name
        self.batch_size = 32
        action_size = 4
    def _build_model(self):
        with keras.backend.get_session().graph.as_default():
            model = Sequential()
            model.add(Dense(512, input_dim=self.state_size, activation='relu'))
            model.add(Dense(256, activation='relu'))
            model.add(Dense(4,activation='linear'))
            # model.add(Dense(4, activation='softmax'))
            model.compile(loss='mse',
                          optimizer=Adam(lr=self.learning_rate))
            return model

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def act(self, state):
        if np.random.rand() <= self.epsilon:  # if acting randomly, take random action
            action = random.randrange(self.action_size)
            # print(action)
            return action
            # return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        # print("this is the predited one: {}".format(act_values))
        # print("act: {}".format(
        #     np.argmax(act_values[0])))  # if not acting randomly, predict reward value based on current state
        return np.argmax(act_values[0])  # pick the action that will give the highest reward (i.e., go left or right?)

    def replay(self):
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state in minibatch:
            target = reward
            reshaped_next_state = np.reshape(next_state.state,[1,self.state_size])
            reshaped_state = np.reshape(state.state,[1,self.state_size])
            # print("old: {},{}  new:{},{}".format(state.hero.current_cell.row,state.hero.current_cell.column,next_state.hero.current_cell.row,next_state.hero.current_cell.column))
            with keras.backend.get_session().graph.as_default():
                target = (reward + self.gamma * np.amax(self.model.predict(reshaped_next_state)[0]))
                target_f = self.model.predict(reshaped_state)
                target_f[0][action] = target
                self.model.fit(reshaped_state, target_f, epochs=1,verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    def load(self):
        self.model.load_weights(self.name)

    def save(self):
        self.model.save_weights(self.name)