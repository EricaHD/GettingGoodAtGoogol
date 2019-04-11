from collections import defaultdict
import random

import numpy as np

class BasicAgent:
    """Agent Architecture"""
    def __init__(self):
        #0 - stop, 1 - switch
        self.Q = defaultdict(lambda: {0:0, 1:0})
        
        #Save results
        self.final_state, self.rewards = [], []
        self.wins = 0

    def getAction(self, s, v):
        
        return random.randint(0, 1)
    
    def gameReset(self):
        pass
    
    def historyReset(self):
        self.final_state, self.rewards = [], []
        self.wins = 0
    
#########################################################################################
#########################################################################################

class SarsaAgent(BasicAgent):
    """SARSA Learning Agent"""
    def __init__(self, alpha, gamma, eps, eps_decay, s_cost):
        super().__init__()
        
        self.alpha, self.gamma, self.eps, self.eps_decay, self.s_cost = alpha, gamma, eps, eps_decay, s_cost
        self.orig_eps = eps
    
        #Q-keys
        self.Q_max = 0
        self.Q_key = lambda s: "{}_{}".format(s, self.Q_max)
        
    def update(self, s, a, s_, a_, r):
        #Get keys
        s_key = self.Q_key(s)
        s__key = self.Q_key(s_)
        
        #Game not finished
        if s_ is not None:
            self.Q[s_key][a] += self.alpha * (r + self.gamma * self.Q[s__key][a_] - self.Q[s_key][a] - self.s_cost)
        else:
            self.Q[s_key][a] += self.alpha * (r - self.Q[s_key][a])
            
    def getAction(self, s, v):
        #Get Q-key
        self.Q_max = self.Q_max if self.Q_max >= v else v
        s_key = self.Q_key(s)
        
        #Epsilon Greedy Strategy
        if random.random() < self.eps:
            action =  random.randint(0, 1)
        else:
            action = max(self.Q[s_key], key=self.Q[s_key].get)
        
        #epsilon decay
        self.eps *= (1 - self.eps_decay)
        return action
            
    def gameReset(self):
        self.Q_max = 0
        self.eps = self.orig_eps
            
#########################################################################################
#########################################################################################
            
class QAgent(BasicAgent):
    """Q-learning Agent"""
    def __init__(self, alpha, gamma, eps, eps_decay, s_cost):
        super().__init__()
        
        self.alpha, self.gamma, self.eps, self.eps_decay, self.s_cost = alpha, gamma, eps, eps_decay, s_cost
        self.orig_eps = eps
    
        #Q-keys
        self.Q_max = 0
        self.Q_key = lambda s: "{}_{}".format(s, self.Q_max)
        
    def update(self, s, a, s_, a_, r):
        #Get keys
        s_key = self.Q_key(s)
        s__key = self.Q_key(s_)
        
        #Game not finished
        if s_ is not None:
            self.Q[s_key][a] += self.alpha * (r + self.gamma * max([self.Q[s__key][a2] for a2 in range(2)]) - self.Q[s_key][a] - self.s_cost)
        else:
            self.Q[s_key][a] += self.alpha * (r - self.Q[s_key][a])
            
    def getAction(self, s, v):
        #Get Q-key
        self.Q_max = self.Q_max if self.Q_max >= v else v
        s_key = self.Q_key(s)
        
        #Epsilon Greedy Strategy
        if random.random() < self.eps:
            action =  random.randint(0, 1)
        else:
            action = max(self.Q[s_key], key=self.Q[s_key].get)
        
        #epsilon decay
        self.eps *= (1 - self.eps_decay)
        return action
            
    def gameReset(self):
        self.Q_max = 0
        self.eps = self.orig_eps
            
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
    
    def gameReset(self):
        self.Q_max = 0
        