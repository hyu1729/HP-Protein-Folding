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
    0: np.array([-1, 1]),
    1: np.array([-1, -1]),
    2: np.array([1, -1]),
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
        self.reward = 0
        self.grid_length = 201
        self.midpoint = (100, 100)
        self.grid = np.zeros(shape = (self.grid_length, self.grid_length), dtype = int)
        
        self.state = np.full((2, len(seq)), 100) #state[0] = x-coords, state[1] = y-coords, state[:,i] = coords for mol i
        for i in range(len(seq)):
            self.state[1][i] = 100 - len(seq) // 2 + i
            
        self.prev_state = np.copy(self.state)
        
        for i in range(len(seq)):
            self.grid[self.state[0, i]][self.state[1, i]] = POLY_TO_INT[self.seq[i]]

        self.action_space = spaces.Discrete(4 * len(seq) + 8 + 1)
        self.observation_space = spaces.Box(low = -1, high = 1, shape = (self.grid_length, self.grid_length,), dtype = int)
        
    def step(self, action):
        if not self.valid(action):
            print("{} is an illegal move".format(action))
            return (False, self.grid, self.reward, False, self.actions)
        
        self.actions.append(action)
        np.copyto(self.prev_state, self.state)
        done = False
        if action == 4 * len(self.seq) + 8: #Stop
            done = True
            return (self.grid, self.reward, done, self.actions)
        elif action >= 4 * len(self.seq) + 4: #For the last mol
            if action - 4 * len(self.seq) - 4 == 3:
                self.state[1][-1] += 2
                self.state[0][-2] = self.state[0][-1]
                self.state[1][-2] = self.state[1][-1] - 1
            elif action - 4 * len(self.seq) - 4 == 2:
                self.state[0][-1] += 2
                self.state[0][-2] = self.state[0][-1] - 1
                self.state[1][-2] = self.state[1][-1]
            elif action - 4 * len(self.seq) - 4 == 1:
                self.state[1][-1] -= 2
                self.state[0][-2] = self.state[0][-1]
                self.state[1][-2] = self.state[1][-1] + 1
            else:
                self.state[0][-1] -= 2
                self.state[0][-2] = self.state[0][-1] + 1
                self.state[1][-2] = self.state[1][-1]
                
            i = len(self.seq) - 3
            self.state[:, i] = self.prev_state[:, i + 2]
            while not self.is_adj(self.state[:, i], self.state[:, i - 1]) and i >= 1:
                i -= 1
                self.state[:,  i] = self.prev_state[:, i + 2]
                
        elif action >= 4 * len(self.seq): #For the first mol
            if action - 4 * len(self.seq) == 3:
                self.state[1][0] += 2
                self.state[0][1] = self.state[0][0]
                self.state[1][1] = self.state[1][0] - 1
            elif action - 4 * len(self.seq) == 2:
                self.state[0][0] += 2
                self.state[0][1] = self.state[0][0] - 1
                self.state[1][1] = self.state[1][0]
            elif action - 4 * len(self.seq) == 1:
                self.state[1][0] -= 2
                self.state[0][1] = self.state[0][0]
                self.state[1][1] = self.state[1][0] + 1
            else:
                self.state[0][0] -= 2
                self.state[0][1] = self.state[0][0] + 1
                self.state[1][1] = self.state[1][0]
                
            i = 2
            self.state[:, i] = self.prev_state[:, i - 2]
            while i <= len(self.seq) - 2 and not self.is_adj(self.state[:, i], self.state[:, i + 1]):
                i += 1
                self.state[:, i] = self.prev_state[:, i - 2]
                
        else: #For all other mols    
            i = action // 4
            dir = action % 4
            loc = DIR_TO_ARRAY[dir] + self.prev_state[:, i]
            self.state[:, i] = loc
            pre = True if i == len(self.seq) - 1 else False
            post = True if i == 0 else False
            if i != 0 and i != len(self.seq) - 1:
                pre = True if self.is_adj(loc, self.state[:, i - 1]) else False # i - 1 adjacent to i's new location
                post = True if self.is_adj(loc, self.state[:, i + 1]) else False # i + 1 adjacent to i's new location
            if pre and post:
                self.grid[self.prev_state[0, i], self.prev_state[1, i]] = 0
                self.grid[self.state[0, i], self.state[1, i]] = POLY_TO_INT[self.seq[i]]
                return (self.grid, self.reward, done, self.actions)
            elif pre:
                C = loc + self.prev_state[:, i] - self.prev_state[:, i - 1]
                if i != len(self.seq) - 1:
                    self.state[:, i + 1] = C
                i += 2
                if i <= len(self.seq) - 1:
                    self.state[:, i] = self.prev_state[:, i - 2]
                    while i <= len(self.seq) - 2 and not self.is_adj(self.state[:, i], self.state[:, i + 1]):
                        i += 1
                        self.state[:, i] = self.prev_state[:, i - 2]

            elif post:
                C = loc + self.prev_state[:, i] - self.prev_state[:, i + 1]
                if i != 0:
                    self.state[:, i - 1] = C
                i -= 2
                if i >= 0:
                    self.state[:, i] = self.prev_state[:, i + 2]
                    while not self.is_adj(self.state[:, i], self.state[:, i - 1]) and i >= 1:
                        i -= 1
                        self.state[:,  i] = self.prev_state[:, i + 2]
            else:
                print("Illegal Move")
                return (False, self.grid, self.reward, done, self.actions)
            
        self.grid = self.update_grid()
        self.reward = self.compute_reward()
        return (True, self.grid, self.reward, done, self.actions)
        
    def compute_reward(self):
        state = []
        for i in range(len(self.seq)):
            if self.seq[i] == 'H':
                state.append((self.state[0][i], self.state[1][i]))
            else:
                state.append((-1000, -1000))
        distances = euclidean_distances(state, state)
        ## We can extract the H-bonded pairs by looking at the upper-triangular (triu)
        ## distance matrix, and taking those = 1, but ignore immediate neighbors (k=2).
        bond_idx = np.where(np.triu(distances, k=2) == 1.0)    
        return len(bond_idx[0])
    
    def is_adj(self, x, y):
        return True if abs(np.sum(x[0] - y[0])) + abs(np.sum(x[1] - y[1])) == 1 else False
    
    def update_grid(self):
        self.grid = np.zeros(shape = (self.grid_length, self.grid_length), dtype = int)
        for i in range(len(self.seq)):
            self.grid[self.state[0, i]][self.state[1, i]] = POLY_TO_INT[self.seq[i]]
        return self.grid        
    
    def valid(self, action):
        if action > 4 * len(self.seq) + 8:
            return False
        elif action == 4 * len(self.seq) + 8: 
            return True
        elif action == 4 * len(self.seq) + 4 + 3:
            return self.grid[self.state[0, -1]][self.state[1, -1] + 2] == 0 and self.grid[self.state[0, -1]][self.state[1, -1] + 1] == 0
        elif action == 4 * len(self.seq) + 4 + 2:
            return self.grid[self.state[0, -1] + 2][self.state[1, -1]] == 0 and self.grid[self.state[0, -1] + 1][self.state[1, -1]] == 0
        elif action == 4 * len(self.seq) + 4 + 1:
            return self.grid[self.state[0, -1]][self.state[1, -1] - 2] == 0 and self.grid[self.state[0, -1]][self.state[1, -1] - 1] == 0
        elif action == 4 * len(self.seq) + 4 :
            return self.grid[self.state[0, -1] - 2][self.state[1, -1]] == 0 and self.grid[self.state[0, -1] - 1][self.state[1, -1]] == 0
        elif action == 4 * len(self.seq) + 3:
            return self.grid[self.state[0, 0]][self.state[1, 0] + 2] == 0 and self.grid[self.state[0, 0]][self.state[1, 0] + 1] == 0
        elif action == 4 * len(self.seq) + 2:
            return self.grid[self.state[0, 0] + 2][self.state[1, 0]] == 0 and self.grid[self.state[0, 0] + 1][self.state[1, 0]] == 0
        elif action == 4 * len(self.seq) + 1:
            return self.grid[self.state[0, 0]][self.state[1, 0] - 2] == 0 and self.grid[self.state[0, 0]][self.state[1, 0] - 1] == 0
        elif action == 4 * len(self.seq):
            return self.grid[self.state[0, 0] - 2][self.state[1, 0]] == 0 and self.grid[self.state[0, 0] - 1][self.state[1, 0]] == 0
        else:
            i = action // 4
            dir = action % 4
            loc = DIR_TO_ARRAY[dir] + self.prev_state[:, i]
            if self.grid[loc[0]][loc[1]] != 0:
                return False
            if i == 0:
                return self.is_adj(loc, self.state[:, 1])
            elif i == len(self.seq) - 1:
                return self.is_adj(loc, self.state[:, -2])
            else:
                return self.is_adj(loc, self.state[:, i - 1]) or self.is_adj(loc, self.state[:, i + 1])

    def reset(self):
        self.actions = []
        self.reward = 0
        self.state = np.full((2, len(self.seq)), 100) #state[0] = x-coords, state[1] = y-coords, state[:,i] = coords for mol i
        for i in range(len(self.seq)):
            self.state[1][i] = 100 - len(self.seq) // 2 + i
            
        self.prev_state = np.copy(self.state)
        return self.grid
    
    def render(self):
        ''' Renders the environment '''
        # Set up plot
        
        state_dict = OrderedDict()
        for i in range(len(self.seq)):
            state_dict.update({ (int(self.state[1][i]) - 100, 100 - int(self.state[0][i])) : self.seq[i] })
        fig, ax = plt.subplots()
        plt.axis('scaled')
        if len(self.actions) != 0:
            plt.title(self.actions[-1])
        bd = 7
        ax.set_xlim([-0.5 - bd, 0.5 + bd])
        ax.set_ylim([-0.5 - bd, 0.5 + bd])
        
        # Plot chain
        dictlist = list(state_dict.items())
        curr_state = dictlist[0]
        mol = plt.Circle(curr_state[0], 0.2, color = 'green' if curr_state[1] == 'H' else 'gray', zorder = 1)
        ax.add_artist(mol)
        mol = plt.Circle(curr_state[0], 0.2, color = 'blue', fill = False, zorder = 2)
        ax.add_artist(mol)
        for i in range(1, len(dictlist)):
            next_state = dictlist[i]
            xdata = [curr_state[0][0], next_state[0][0]]
            ydata = [curr_state[0][1], next_state[0][1]]
            bond = mlines.Line2D(xdata, ydata, color = 'k', zorder = 0)
            ax.add_line(bond)
            mol = plt.Circle(next_state[0], 0.2, color = 'green' if next_state[1] == 'H' else 'gray', zorder = 1)
            ax.add_artist(mol)
            curr_state = next_state
        
        # Show H-H bonds
        ## Compute all pair distances for the bases in the configuration
        state = []
        for i in range(len(dictlist)):
            if dictlist[i][1] == 'H':
                state.append(dictlist[i][0])
            else:
                state.append((-1000, 1000)) #To get rid of P's
        distances = euclidean_distances(state, state)
        ## We can extract the H-bonded pairs by looking at the upper-triangular (triu)
        ## distance matrix, and taking those = 1, but ignore immediate neighbors (k=2).
        bond_idx = np.where(np.triu(distances, k=2) == 1.0)    
        for (x,y) in zip(*bond_idx):
            xdata = [state[x][0], state[y][0]]
            ydata = [state[x][1], state[y][1]]
            backbone = mlines.Line2D(xdata, ydata, color = 'r', ls = ':', zorder = 1)
            ax.add_line(backbone)