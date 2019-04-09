from collections import defaultdict
import random

import numpy as np

class BasicAgent:
    def __init__(self, alpha, gamma, eps, eps_decay, s_cost):
        self.alpha, self.gamma, self.eps, self.eps_decay, self.s_cost = alpha, gamma, eps, eps_decay, s_cost
        
        self.actions = [0, 1]
        #0 - stop, 1 - switch
        self.Q = defaultdict(lambda: {0:0, 1:0})
        self.Q[0] = {0:0, 1:0}
        self.final_state, self.rewards, self.wins = [], [], []

    def get_action(self, s):
        #Epsilon Greedy Strategy
        if random.random() < self.eps:
            action =  random.randint(0, 1)
        else:
            action = max(self.Q[s], key=self.Q[s].get)
        
        #epsilon decay
        self.eps *= (1 - self.eps_decay)
        return action

class SarsaAgent(BasicAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def update(self, s, a, s_, a_, r):
        if s_ is not None:
            self.Q[s][a] += self.alpha * (r + self.gamma * self.Q[s_][a_] - self.Q[s][a] - self.s_cost)
        else:
            self.Q[s][a] += self.alpha * (r - self.Q[s][a])
            
class QAgent(BasicAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def update(self, s, a, s_, a_, r):
        if s_ is not None:
            self.Q[s][a] += self.alpha * (r + self.gamma * max([self.Q[s_][a2] for a2 in range(2)]) - self.Q[s][a] - self.s_cost)
        else:
            self.Q[s][a] += self.alpha * (r - self.Q[s][a])
            
        
        