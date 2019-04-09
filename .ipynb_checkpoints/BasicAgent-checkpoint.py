from collections import defaultdict
import random

import numpy as np

class BasicAgent:
    def __init__(self, alpha, gamma, eps, eps_decay, s_cost):
        self.alpha, self.gamma, self.eps, self.eps_decay, self.s_cost = alpha, gamma, eps, eps_decay, s_cost
        
        self.actions = [0, 1]
        #0 - stop, 1 - switch
        self.Q = defaultdict(lambda: {0:0, 1:0.1})
        self.Q[0] = {0:0, 1:0}
        self.rewards = []
        self.final_state = []

    def get_action(self, s):
        #Epsilon Greedy Strategy
        if random.random() < self.eps:
            action =  random.randint(0, 1)
        else:
            action = max(self.Q[s], key=self.Q[s].get)
        
        #epsilon decay
        self.eps *= (1 - self.eps_decay)
        return action
        