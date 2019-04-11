from collections import defaultdict
import random

import numpy as np

class BasicAgent:
    """Agent Architecture"""
    def __init__(self):
        #0 - stop, 1 - switch
        self.Q = defaultdict(lambda: {0:0, 1:0})

    def getAction(self, s, v):
        
        return random.randint(0, 1)
    
    def update(self):
        pass
    
    def resetGame(self):
        pass

#########################################################################################
#########################################################################################
            
class QAgent(BasicAgent):
    """Q-learning Agent"""
    def __init__(self, alpha, gamma, eps, eps_decay, s_cost, sarsa, Q_key_fn, Q_key):
        super().__init__()
        
        self.alpha, self.gamma, self.eps, self.eps_decay, self.s_cost = alpha, gamma, eps, eps_decay, s_cost
        self.orig_eps = eps
        
        self.sarsa = sarsa
        
        self.Q_key_fn = Q_key_fn
        self.Q_key, self.orig_Q_key = Q_key, Q_key
    
    def getAction(self, s, v):
        #Get Q-key
        self.Q_key = self.Q_key_fn(s, v, self.Q_key)
        
        #Epsilon Greedy Strategy
        if random.random() < self.eps:
            action =  random.randint(0, 1)
        else:
            action = max(self.Q[self.Q_key], key=self.Q[self.Q_key].get)
        
        #epsilon decay
        self.eps *= (1 - self.eps_decay)
        return action
        
    def update(self, s, a, s_, a_, r):
        #Get keys
        s_key = self.Q_key_fn(s, None, self.Q_key)
        s__key = self.Q_key_fn(s_, None, self.Q_key)
        
        #Game not finished
        if s_ is not None:
            #SARSA update
            if self.sarsa:
                self.Q[s_key][a] += self.alpha * (r + self.gamma * self.Q[s__key][a_] - self.Q[s_key][a] - self.s_cost)
            else:
                self.Q[s_key][a] += self.alpha * (r + self.gamma * max([self.Q[s__key][a2] for a2 in range(2)]) - self.Q[s_key][a] - self.s_cost)
        else:
            self.Q[s_key][a] += self.alpha * (r - self.Q[s_key][a])
            
    def observe(self, s, v):
        #Add key to Q
        self.Q_key = self.Q_key_fn(s, v, self.Q_key)
        self.Q[self.Q_key]
            
    def resetGame(self):
        self.Q_max = 0
        self.eps = self.orig_eps
        self.Q_key = self.orig_Q_key
            
#########################################################################################
#########################################################################################

class OptimalAgent(BasicAgent):
    """Optimal E Agent"""
    def __init__(self, n_states):
        super().__init__()
        
        self.euler = 1/np.exp(1)
        
        self.n_states = n_states
        self.g_state = int(self.euler * n_states)
        
        self.Q_max = 0
        
    def update(self, s, a, s_, a_, r):
        pass
    
    def getAction(self, s, v):
        if s <= self.g_state:
            self.Q_max = self.Q_max if self.Q_max >= v else v
            return 1
        else:
            if self.Q_max < v:
                return 0
            else:
                return 1
    
    def resetGame(self):
        self.Q_max = 0
        