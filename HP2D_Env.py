import numpy as np

ACTION_TO_ARRAY = {
    0 : np.array([-1, 0]),
    1 : np.array([0, 1]),
    2 : np.array([0, -1]),
    3 : np.array([1, 0])
}

POLY_TO_INT = {
    'H' : 1, 'P' : -1
}

class HP2D():
    '''
    2D Lattice Environment for AGZ MCTS
    '''
    
    def __init__(self, seq, shape):
        self.seq = seq
        self.shape = shape
        
    def make_state(self):
        state = np.zeros(self.shape)
        ##FINISH
        return state
    
    def get_pos(self, state):
        if (state[0] == state[4]).all():
            return np.argwhere(np.array([state[1] != state[5]]) == True)[0]
        else:
            return np.argwhere(np.array([state[0] != state[4]]) == True)[0]     
    
    def stringrep(self, state):
        ''' 
        String representation of state
        Represent state by actions taken to get there
        '''
        actions = []
        actions_str = [str(a) for a in actions]
        return ''.join(actions_str)
    
    def done(self, state):
        '''
        Return 0 if not ended, 1 if done
        '''
        return 1 if state[8][0][0] == 0 else 0
        
    def valid_moves(self, state):
        last_pos = np.array(self.get_pos(state)[1:])
        vm = [0, 0, 0, 0]
        for a in range(4):
            if is_valid(last_pos + ACTION_TO_ARRAY[a], state):
                vm[a] = 1
        return a
    
    def is_valid(self, pos, state):
        '''
        Check if pos is a valid position
        '''
        if pos not in np.argwhere(np.zeros(state[0].shape) == 0)[0]:
            return False
        if state[0][pos[0]][pos[1]] != 0:
            return False
        if state[1][pos[0]][pos[1]] != 0:
            return False
        return True
        
    def next_state(self, state, action):
        ##FINISH
        num_mol = np.count_nonzero(state[0]) + np.count_nonzero(state[1])
        if num_mol + len(state) - 8 < len(self.seq):
            next_mol = POLY_TO_INT[seq[num_mol + 4]]
        else:
            next_mol = 0
        ns = state
        ns[4:8] = state[0:4]
        ns[8:-1] = state[9:]
        ns[-1] = np.full(state[0].shape, next_mol)
        add_mol = int(state[8][0][0])
        last_pos = np.array(self.get_pos(state)[1:])
        next_pos = last_pos + ACTION_TO_ARRAY[action]
        ns[(1 - add_mol) // 2][next_pos[0]][next_pos[1]] = 1
        ### Fill out C_t, B_t
        return ns