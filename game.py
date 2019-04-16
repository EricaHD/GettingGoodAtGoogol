import numpy as np
from tqdm import tqdm

class Game():
    def __init__(self, lo, hi, n_states, replace):    
        self.lo = lo
        self.hi = hi
        self.n_states = n_states
        self.replace = replace
        
        self.reset()
    
    def step(self):
        self.state += 1
        self.val = self.states[self.state]
        return self.val
      
    def winState(self):
        if self.states[self.state] == self.max_val:
            return True
        else:
            return False
        
    def reset(self):
        """Reset the game states """
        self.states = np.random.choice(np.arange(self.lo, self.hi+1), 
                                       size=self.n_states, 
                                       replace=self.replace)
        self.states_sorted = np.sort(self.states) 
        
        self.state = 0
        self.val = self.states[0]
        
        self.max_val = self.states.max()
        self.max_state = self.states.argmax()
        
    