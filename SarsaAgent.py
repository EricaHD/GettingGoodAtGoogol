from collections import defaultdict
import random

import numpy as np

from BasicAgent import BasicAgent

class SarsaAgent(BasicAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def update(self, s, a, s_, a_, r):
        if s_ is not None:
            self.Q[s][a] += self.alpha * (r + self.gamma * self.Q[s_][a_] - self.Q[s][a] - self.s_cost)
        else:
            self.Q[s][a] += self.alpha * (r - self.Q[s][a])
        
        