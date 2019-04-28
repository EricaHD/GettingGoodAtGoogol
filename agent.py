from collections import defaultdict
import random

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        self.q_key, self.prev_q_key, self.orig_q_key = "", "", ""
        self.q_key_params = q_key_params
    
    def getAction(self, params):
        if (params['val'] == params['hi']) or (params['idx'] == params['n_idx']-1):
            return 0
        
        #Get Q-key
        self.v_key = self.v_fn(params, params['val'], self.v_key)
        q_key = self.q_key_fn(params, params['idx'], self.v_key, self.q_key_params)
        
        #Epsilon Greedy in Train Mode
        if self.mode == "Train":
            #Epsilon
            if random.random() < self.eps:
                action =  random.randint(0, 1)
            else:
                #Equal
                if self.Q[q_key][0] == self.Q[q_key][1]:
                    action =  random.randint(0, 1)
                #Greedy
                else:
                    action = max(self.Q[q_key], key=self.Q[q_key].get)
            
            self.eps *= (1 - self.eps_decay)
            
            self.prev_q_key = self.q_key
            self.q_key = q_key
            return action      
        else:
            #Equal
            if self.Q[q_key][0] == self.Q[q_key][1]:
                action =  random.randint(0, 1)
            #Greedy
            else:
                action = max(self.Q[q_key], key=self.Q[q_key].get)
            
            return action

    def update(self, params):
            
        #Middle of game update
        if not params['game_status']:
            #Get keys
            q_key, a, r  = self.prev_q_key, 1, params['reward']
            q__key, a_ = self.q_key, 0
            
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
        self.v_key = self.orig_v_key
        self.q_key, self.prev_q_key = self.orig_q_key, self.orig_q_key
        
            
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
        
#########################################################################################
#########################################################################################
    
class DQAgent(BasicAgent):
    """DQN Agent"""
    def __init__(self, batch_size, gamma, eps, eps_decay, target_update, p_to_s, p_net, t_net, optimizer, loss, memory, v_fn, v_key, device):
        super().__init__()
        
        self.batch_size, self.gamma, self.eps, self.eps_decay, self.target_update = batch_size, gamma, eps, eps_decay, target_update
        self.eps_orig = eps
        
        self.p_to_s = p_to_s
        
        self.policy_net, self.target_net = p_net, t_net
        self.optimizer, self.loss = optimizer, loss
        
        self.memory = memory
        
        self.v_fn, self.v_key, self.orig_v_key = v_fn, v_key, v_key
    
	self.device = device

    def getAction(self, params):
        
        #Get new v
        self.v_key = self.v_fn(params, params['val'], self.v_key)
        
        state = self.p_to_s(params, self.v_key).to(self.device)
        
        if self.mode == "Train":
            if (params['val'] == params['hi']) or (params['idx'] == params['n_idx']-1):
                action = torch.tensor([[0.]], device=self.device, dtype=torch.long)
            elif random.random() < self.eps:
                action = torch.tensor([[random.randrange(2)]], device=self.device, dtype=torch.long)
            else:
                with torch.no_grad():
                    action = self.policy_net(state).max(1)[1].view(1, 1).to(self.device)
            
            self.eps *= (1 - self.eps_decay)
            return action, state
        else:
            if (params['val'] == params['hi']) or (params['idx'] == params['n_idx']-1):
                action = torch.tensor([[0.]], device=self.device, dtype=torch.long)
            else:
                with torch.no_grad():
                    action = self.policy_net(state).max(1)[1].view(1, 1).to(self.device)
            return action
                
    def update(self):
        if len(self.memory) < self.batch_size:
            return
        
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
    
        #Compute masks for non-final games
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=self.device, dtype=torch.uint8)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        
        #Sep state, action, reward
        state_batch = torch.cat(batch.state).to(self.device)
        action_batch = torch.cat(batch.action).to(self.device)
        reward_batch = torch.cat(batch.reward).to(self.device)
    
        #Q(s_t, a)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch).to(self.device)

        #V(s_{t+1})
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
    
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Loss
        loss = self.loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
    def train(self):
        self.mode="Train"
        self.policy_net.train()
        self.target_net.eval()
        
    def eval(self):
        self.mode="Eval"
        self.policy_net.eval()
    
    def updateNet(self, game_i):
        if game_i % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            
    def reset(self):
        self.v_key = self.orig_v_key
        self.optimizer.zero_grad()
