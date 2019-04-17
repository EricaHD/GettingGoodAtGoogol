import numpy as np
from tqdm import tqdm

class Game():
    def __init__(self, lo, hi, n_idx, replace, reward_fn):    
        self.lo = lo
        self.hi = hi
        self.n_idx = n_idx
        self.replace = replace
        self.reward_fn = reward_fn
        
        self.reset()
    
    def step(self):
        self.idx += 1
        self.val = self.values[self.idx]
        
        return self.idx, self.val
    
    def getReward(self):
        self.setGameStatus()
        return self.reward_fn(self)
      
    def checkWin(self):
        if self.values[self.idx] == self.max_val:
            return True
        else:
            return False
        
    def setGameStatus(self):
        self.game_over = True
        
    def reset(self):
        """Reset the game"""
        self.values = np.random.choice(np.arange(self.lo, self.hi+1), 
                                       size=self.n_idx, 
                                       replace=self.replace)
        self.values_sorted = np.sort(self.values) [::-1]
        
        self.idx = 0
        self.val = self.values[0]
        
        self.max_val = self.values.max()
        self.max_idx = self.values.argmax()
        
        self.game_over = False
        
    