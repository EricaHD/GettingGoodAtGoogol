from collections import defaultdict
import random

import numpy as np

class BasicAgent:
    def __init__(self, alpha, gamma, eps, eps_decay, s_cost):
        self.alpha, self.gamma, self.eps, self.eps_decay, self.s_cost = alpha, gamma, eps, eps_decay, s_cost
        self.orig_eps = eps
        
        #0 - stop, 1 - switch
        self.Q = defaultdict(lambda: {0:0, 1:0})
        
        #Save results
        self.final_state, self.rewards = [], []
        self.wins = 0
        
        #Q-keys
        self.Q_max = 0
        self.Q_key = lambda s: "{}_{}".format(s, self.Q_max)

    def get_action(self, s, v):
        
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
    
    def reset(self):
        self.Q_max = 0
        self.eps = self.orig_eps
    
#########################################################################################
#########################################################################################

class SarsaAgent(BasicAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def update(self, s, a, s_, a_, r):
        #Get keys
        s_key = self.Q_key(s)
        s__key = self.Q_key(s_)
        
        #Game not finished
        if s_ is not None:
            self.Q[s_key][a] += self.alpha * (r + self.gamma * self.Q[s__key][a_] - self.Q[s_key][a] - self.s_cost)
        else:
            self.Q[s_key][a] += self.alpha * (r - self.Q[s_key][a])
            
#########################################################################################
#########################################################################################
            
class QAgent(BasicAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def update(self, s, a, s_, a_, r):
        #Get keys
        s_key = self.Q_key(s)
        s__key = self.Q_key(s_)
        
        #Game not finished
        if s_ is not None:
            self.Q[s_key][a] += self.alpha * (r + self.gamma * max([self.Q[s__key][a2] for a2 in range(2)]) - self.Q[s_key][a] - self.s_cost)
        else:
            self.Q[s_key][a] += self.alpha * (r - self.Q[s_key][a])
            
        
        