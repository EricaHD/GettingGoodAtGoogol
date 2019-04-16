from collections import defaultdict
import random

import numpy as np

class BasicAgent:
    """Agent Architecture"""
    def __init__(self):
        pass

    def getAction(self, params):
        return random.randint(0, 1)
    
    def update(self, params):
        pass
    
    def reset(self):
        pass

#########################################################################################
#########################################################################################
            
class QAgent(BasicAgent):
    """Q-learning Agent"""
    def __init__(self, alpha, gamma, eps, eps_decay, s_cost, sarsa, q_key_fn, q_key):
        super().__init__()
        
        self.alpha, self.gamma, self.eps, self.eps_decay, self.s_cost = alpha, gamma, eps, eps_decay, s_cost
        self.orig_eps = eps
        
        self.Q = defaultdict(lambda: {0:0, 1:0})
        
        self.sarsa = sarsa
        
        self.q_key_fn = q_key_fn
        self.q_key, self.orig_q_key = q_key, q_key
    
    def getAction(self, params, update_key=True):
        
        #Get Q-key
        s_key = self.q_key_fn(params, self.q_key)
        
        if (params['val'] == params['hi']) or (params['state'] == params['n_states']-1):
            action = 0
        else:
            #Epsilon Greedy Strategy
            if random.random() < self.eps:
                action =  random.randint(0, 1)
            else:
                if self.Q[s_key][0] == self.Q[s_key][1]:
                    action =  random.randint(0, 1)
                else:
                    action = max(self.Q[s_key], key=self.Q[s_key].get)

        #epsilon decay
        self.eps *= (1 - self.eps_decay)
        
        if update_key:
            self.q_key = s_key
            return action
        else:
            return s_key, action
        
    def update(self, params):
            
        if not params['game_over']:
            s_key, a, r  = self.q_key, 1, params['reward'] 
            s__key, a_ = self.getAction(params=params, update_key=False)
            
            if self.sarsa:
                self.Q[s_key][a] += self.alpha * (r + self.gamma * self.Q[s__key][a_] - self.Q[s_key][a] - self.s_cost)
            else:
                self.Q[s_key][a] += self.alpha * (r + self.gamma * max([self.Q[s__key][a2] for a2 in range(2)]) - self.Q[s_key][a] - self.s_cost)
        else:
            self.Q[self.q_key][0] += self.alpha * (params['reward'] - self.Q[self.q_key][0] - self.s_cost)
            
    def reset(self):
        self.Q_max = 0
        self.eps = self.orig_eps
        self.q_key = self.orig_q_key
        
            
#########################################################################################
#########################################################################################

class OptimalAgent(BasicAgent):
    """Optimal E Agent"""
    def __init__(self, n_states, max_val):
        super().__init__()
        
        self.euler = 1/np.exp(1)
        
        self.n_states = n_states
        self.g_state = int(self.euler * n_states)
        
        self.Q_max = 0
        
        self.max_val = max_val
        
    def update(self, s, a, s_, a_, r):
        pass
    
    def getAction(self, params):
        
        if params['val'] == params['hi']:
            return 0
    
        if params['state'] <= self.g_state:
            self.Q_max = self.Q_max if self.Q_max >= params['val'] else params['val']
            return 1
        else:
            if self.Q_max < params['val']:
                return 0
            else:
                return 1
    
    def reset(self):
        self.Q_max = 0
        