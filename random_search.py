import random
from itertools import product
import numpy as np
from tqdm import trange

from trainer import *
from game import Game
from agent import *
from utils import *

##################################################
# SET UP GAME
##################################################
 
game_params = {'lo':1,
               'hi':10000,
               'n_idx':50,
               'replace':False,
               'reward_fn':rewardTopN,
               'reward':{'pos':10, 'neg':-10, 'n':11} }
    
game = Game(**game_params)

trainer = QTrainer()

param_ranges = {
                "alpha":10.**np.arange(-4, 1, 1),
                "alpha_decay":10.**np.arange(-8, -4, 1),
                "gamma":np.arange(.0, 1.05, 0.05),
                "eps":np.arange(.0, .6, 0.05),
                "eps_decay":10.**np.arange(-8, 0, 1),
                "s_cost":[0, 1 , 2, 3],
                "sarsa":[False, True]
               }

param_keys = param_ranges.keys()

param_values = list(product(*param_ranges.values()))

choices = np.random.choice(np.arange(0, len(param_values)), replace=False)

best_params, best_score = None, 0

for i in trange(60):
    params = dict(*zip(param_keys, choices[i]))
    
    agent_params = {'alpha':args['alpha'],
                    'alpha_decay':args['alpha_decay'],
                    'alpha_step':args['alpha_step'],
                    'gamma':args['gamma'],
                    'eps':args['epsilon'], 
                    'eps_decay':args['eps_decay'], 
                    's_cost':args['s_cost'],
                    'sarsa':args['q_learn'],
                    'q_key_fn':q_key_fn,
                    'q_key_params':q_key_params,
                    'v_fn':v_fn,
                    'v_key':v_key}




    
    

    
agent_params = {'alpha':args['alpha'],
                        'alpha_decay':args['alpha_decay'],
                        'alpha_step':args['alpha_step'],
                      	'gamma':args['gamma'],
                      	'eps':args['epsilon'], 
                      	'eps_decay':args['eps_decay'], 
                      	's_cost':args['s_cost'],
                      	'sarsa':args['q_learn'],
                      	'q_key_fn':q_key_fn,
                        'q_key_params':q_key_params,
                      	'v_fn':v_fn,
                        'v_key':v_key}
        
        agent = QAgent(**agent_params)