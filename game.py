import numpy as np
from tqdm import tqdm


class Game:
    def __init__(self, lo, hi, n_idx, replace, reward_fn, reward, dist):
        self.lo = lo
        self.hi = hi
        self.n_idx = n_idx
        self.replace = replace
        self.reward_fn = reward_fn
        self.reward = reward
        self.dist = dist
        
        self.reset()
    
    def step(self, action):  
        
        if action == 0:
            self.setGameStatus()
            reward, self.win = self.reward_fn(self)
            game_over = self.game_over
        else:
            self.idx += 1
            self.val = self.values[self.idx]
            reward, game_over = 0., self.game_over
        
        return self.idx, self.val, float(reward), game_over
    
    def setGameStatus(self):
        self.game_over = True
        
    def reset(self):
        """Reset the game"""
        if self.dist == 'uniform':
            self.values = np.random.choice(np.arange(self.lo, self.hi+1),
                                           size=self.n_idx,
                                           replace=self.replace)
        else:  # self.dist == 'normal'
            self.values = np.zeros(self.n_idx)
            for i in range(self.n_idx):
                self.values[i] = np.round_(np.random.normal(50000, 17000))  # hardcoded mean and std
                while self.values[i] < self.lo or self.values[i] > self.hi:  # keep generating if outside [lo, hi]
                    self.values[i] = np.round_(np.random.normal(50000, 17000))

        self.values_sorted = np.sort(self.values)[::-1]
        
        self.idx = 0
        self.val = self.values[0]
        
        self.max_val = self.values.max()
        self.max_idx = self.values.argmax()
        
        self.game_over = False
        self.win = 0
