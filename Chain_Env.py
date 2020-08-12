import sys
from math import floor
from collections import OrderedDict
from sklearn.metrics.pairwise import euclidean_distances
import itertools

import gym
from gym import (spaces, utils, logger)
import numpy as np
from six import StringIO

import matplotlib.pyplot as plt
import matplotlib.lines as mlines

POLY_TO_INT = {
    'H' : 1, 'P' : -1
}

DIR_TO_ARRAY = {
    0: np.array([1, -1]),
    1: np.array([-1, -1]),
    2: np.array([-1, 1]),
    3: np.array([1, 1])
}

class ChainEnv(gym.Env):
    def __init__(self, seq):
        try:
            if not set(seq.upper()) <= set('HP'):
                raise ValueError("%r (%s) is an invalid sequence" % (seq, type(seq)))
            self.seq = seq.upper()
        except AttributeError:
            logger.error("%r (%s) must be of type 'str'" % (seq, type(seq)))
            raise

        try:
            if len(seq) > 100:
                raise ValueError("%r (%s) must have length <= 100" % (seq, type(seq)))
        except AttributeError:
            logger.error("%r (%s) must be of type 'str'" % (seq, type(seq)))
            raise
            
        self.seq = seq
        self.actions = []
        self.grid_length = 201
        self.midpoint = (100, 100)
        self.grid = np.zeros(shape = (self.grid_length, self.grid_length), dtype = int)
        
        self.state = np.full((2, len(seq)), 100) #state[0] = x-coords, state[1] = y-coords, state[:,i] = coords for mol i
        for i in range(len(seq)):
            self.state[1][i] = 100 - len(seq) // 2 + i
            
        for i in range(len(seq)):
            self.grid[self.state[0, i]][self.state[1, i]] = POLY_TO_INT[self.seq[i]]
            
        print(self.state)
            
        self.action_space = spaces.Discrete(4 * (len(seq) - 2) + 18 + 1)
        self.observation_space = spaces.Box(low = -1, high = 1, shape = (self.grid_length, self.grid_length,), dtype = int)
        
    def step(self, action):
        
        self.actions.append(action)
        done = False
        if action == 4 * len(self.seq) + 18: #Stop
            done = True
        elif action >= 4 * len(self.seq) + 9: #For the last mol
            i = len(self.seq) - 1
            dir = action - 4 * len(self.seq) - 9
        elif action >= 4 * len(self.seq): #For the first mol
            i = 0
            dir = action - 4 * len(self.seq)
        else: #For all other mols    
            i = action // 4 + 1
            dir = action % 4
            temp = np.copy(self.state)
            loc = DIR_TO_ARRAY[dir] + self.state[:, i]
            temp[:, i] = loc
            pre = True if abs(np.sum(self.state[:, i] - self.state[:, i - 1])) + abs(np.sum(loc - self.state[:, i - 1])) == 2 else False
            post = True if abs(np.sum(self.state[:, i] - self.state[:, i + 1])) + abs(np.sum(loc - self.state[:, i + 1])) == 2 else False 
            if pre and post:
                
            elif pre:
                C = loc + self.state[:, i] - self.state[:, i - 1]
                temp[:, i + 1] = C
                temp[:, i + 2:] = self.state[:, i:-2]
                
            elif post:
                C = loc + self.state[:, i + 1] - self.state[:, i]
                temp[:, i - 1] = C
                temp[:, :i - 1] = self.state[:, 2:i + 1]
            print(temp)
            
        
        reward = self.compute_reward()
        return (self.grid[97:102, 97:102], reward, done, self.actions)
            
    def compute_reward(self):
        return 0