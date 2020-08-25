import numpy as np
from Chain_Env.py import ChainEnv

class Chain():
    
    def __init__(self, seq, max_len, grid_len):
        self.seq = seq
        self.env = ChainEnv(seq)
        self.max_len = max_len
        self.grid_len = grid_len
        
    def make_state(self):
        