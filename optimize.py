import random
from itertools import product
import numpy as np
from scipy.optimize import fmin
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

agent_params = {'alpha':0.001,
                'alpha_decay':0.00001,
                'alpha_step':100,
                'gamma':0.8,
                'eps':0.1, 
                'eps_decay':0.0001, 
                's_cost':0,
                'sarsa':False,
                'q_key_fn':qKeyMaxBin,
                'q_key_params':2_2,
                'v_fn':vMax,
                'v_key':-1
               }
agent = QAgent(**agent_params)

#Parameters to optimize
keys = ['alpha', 'alpha_decay', 'gamma', 'eps', 'eps_decay']

#Curriculum
curr_params = {"epoch":1000, 
               'params':{"op":"-",
                           "n":1
                                    }
              }

trainer = QTrainer()

def optimizeAgent(params):
    
    agent.reset()
    
    for i, key in enumerate(keys):
        agent[key] = params[i]
        
    score = 1 - trainer.train(game, agent, 10_000, 0, 0, curr)
    return score

params = fmin(optimizeAgent, (0.001, 0.00001, 0.9, 0.1, 0.0001))

for i, key in enumerate(keys):
    print("{}: {}".format(key, params[i]))

    
    
