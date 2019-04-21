from collections import defaultdict
import random

import numpy as np

class BasicAgent:
    """Agent Architecture"""
    def __init__(self):
        self.mode = "Train"
        pass

    def getAction(self, params):
        return random.randint(0, 1)
    
    def update(self, params):
        pass
    
    def reset(self):
        pass
    
    def train(self):
        self.mode = "Train"
        
    def eval(self):
        self.mode = "Eval"

#########################################################################################
#########################################################################################
            
class QAgent(BasicAgent):
    """Q-learning Agent"""
    def __init__(self, alpha, alpha_decay, alpha_step, gamma, eps, eps_decay, s_cost, sarsa, v_fn, v_key, q_key_fn, q_key_params):
        super().__init__()
        
        self.alpha, self.gamma, self.eps, self.eps_decay, self.s_cost = alpha, gamma, eps, eps_decay, s_cost
        self.alpha_decay, self.alpha_step = alpha_decay, alpha_step
        self.orig_eps = eps
        
        self.Q = defaultdict(lambda: {0:0, 1:0})
        
        self.sarsa = sarsa
       
        self.v_fn = v_fn
        self.v_key, self.orig_v_key = v_key, v_key
        
        self.q_key_fn = q_key_fn
        self.q_key, self.orig_q_key = "", ""
        self.q_key_params = q_key_params
    
    def getAction(self, params, update_key=True):
        
        #Get Q-key
        self.v_key = self.v_fn(params, params['val'], self.v_key)
        q_key = self.q_key_fn(params, params['idx'], self.v_key, self.q_key_params)
        
        #Epsilon Greedy in Train Mode
        if self.mode == "Train":
            #If auto-win or auto-stop
            if (params['val'] == params['hi']) or (params['idx'] == params['n_idx']-1):
                action = 0
            #Epsilon
            elif random.random() < self.eps:
                action =  random.randint(0, 1)
            else:
                #Equal
                if self.Q[q_key][0] == self.Q[q_key][1]:
                    action =  random.randint(0, 1)
                #Greedy
                else:
                    action = max(self.Q[q_key], key=self.Q[q_key].get)
            
            self.eps *= (1 - self.eps_decay)
            
            #Called by Update fn
            if update_key: 
                self.q_key = q_key
                return action
            else:
                return q_key, action            
        else:
            #If auto-win or auto-stop
            if (params['val'] == params['hi']) or (params['idx'] == params['n_idx']-1):
                action = 0
            #Equal
            elif self.Q[q_key][0] == self.Q[q_key][1]:
                action =  random.randint(0, 1)
            #Greedy
            else:
                action = max(self.Q[q_key], key=self.Q[q_key].get)
            
            return action

        
    def update(self, params):
            
        #Middle of game update
        if not params['game_over']:
            #Get keys
            q_key, a, r  = self.q_key, 1, params['reward'] 
            q__key, a_ = self.getAction(params=params, update_key=False)
            
            #Update for SARSA
            if self.sarsa:
                self.Q[q_key][a] += self.alpha * (r + self.gamma * self.Q[q_key][a_] - self.Q[q_key][a] - self.s_cost)
            else:
                self.Q[q_key][a] += self.alpha * (r + self.gamma * max([self.Q[q__key][a2] for a2 in range(2)]) - self.Q[q_key][a] - self.s_cost)
        #Game over update
        else:
            self.Q[self.q_key][0] += self.alpha * (params['reward'] - self.Q[self.q_key][0] - self.s_cost)
            
            #Alpha decay
            if params['game_i'] % self.alpha_step == 0:
                self.alpha *= (1 - self.alpha_decay)
            
            
    def reset(self):
        self.eps = self.orig_eps
        self.v_key = self.orig_v_key
        self.q_key = self.orig_q_key
        
            
#########################################################################################
#########################################################################################

class OptimalAgent(BasicAgent):
    """Optimal E Agent"""
    def __init__(self, n_idx, max_val):
        super().__init__()
        
        self.euler = 1/np.exp(1)
        
        self.n_idx = n_idx
        self.g_idx = int(self.euler * n_idx)
        
        self.Q_max = 0
        
        self.max_val = max_val
        
    def getAction(self, params):
        
        if params['val'] == params['hi']:
            return 0
    
        if params['idx'] <= self.g_idx:
            self.Q_max = self.Q_max if self.Q_max >= params['val'] else params['val']
            return 1
        else:
            if self.Q_max < params['val']:
                return 0
            else:
                return 1
    
    def reset(self):
        self.Q_max = 0

#########################################################################################
#########################################################################################

class MCMCAgent(BasicAgent):
    """Monte-Carlo Markov Chain Agent"""
    def __init__(self, gamma, eps, eps_decay, s_cost, v_fn, v_key, q_key_fn, q_key_params):
        super().__init__()
        self.gamma, self.eps, self.eps_decay, self.s_cost = gamma, eps, eps_decay, s_cost
        
        self.v_fn = v_fn
        self.v_key, self.orig_v_key = v_key, v_key
        
        self.Q = defaultdict(lambda: {0:0, 1:0})
        self.q_key_fn = q_key_fn
        self.q_key_params = q_key_params
        
        self.policy = defaultdict(lambda: random.randint(0, 1)) 
        
        self.returns = defaultdict(lambda: {0:0, 1:0})
        self.counts = defaultdict(lambda: {0:0, 1:0})

    
    def getAction(self, params):
        
        #Get new v
        self.v_key = self.v_fn(params, params['val'], self.v_key)
        
        #Training mode is Epsilon-Greedy
        if self.mode == "Train":
            #Auto-win or auto-stop
            if (params['idx'] == params['n_idx']-1) or (self.v_key == params['hi']):
                action = 0
            #Epsilon
            elif random.random() < self.eps:
                action =  random.randint(0, 1)
            #Greedy
            else:
                p_key = self.q_key_fn(params, params['idx'], self.v_key, self.q_key_params)
                action = self.policy[p_key]
            
            #Epsilon Decay
            self.eps *= (1 - self.eps_decay)
            
            return action, self.v_key
        else:
            #Auto-win or auto-stop
            if (params['idx'] == params['n_idx']-1) or (self.v_key == params['hi']):
                action = 0
            else:
                p_key = self.q_key_fn(params, params['idx'], self.v_key, self.q_key_params)
                action = self.policy[p_key]

            return action
    
    def update(self, params, episode):
        
        visited = defaultdict(lambda: {0:False, 1:False})
        
        for e in range(len(episode)):
            idx, val, action, reward = episode[e]
            q_key = self.q_key_fn(params, idx, val, self.q_key_params)
            self.counts[q_key][action] += 1
                
            if not visited[q_key][action]:
                ep_reward = 0
                for j, (idx_, val_, action_, reward_) in enumerate(episode[e:]):
                    ep_reward += ((self.gamma ** j) *  reward_) - self.s_cost

            self.returns[q_key][action] = self.returns[q_key][action] + (ep_reward - self.returns[q_key][action])/self.counts[q_key][action]
            visited[q_key][action] = True
            
        for ret in self.returns.keys():
            if self.returns[ret][0] == self.returns[ret][1]:
                self.policy[ret] = random.randint(0, 1)
            elif self.returns[ret][0] > self.returns[ret][1]:
                self.policy[ret] = 0
            else:
                self.policy[ret] = 1
        
    def reset(self):
        self.v_key = self.orig_v_key
        
    def resetFull(self):
        self.v_key = self.orig_v_key
        self.returns = defaultdict(lambda: {0:0, 1:0})
        self.counts = defaultdict(lambda: {0:0, 1:0})
