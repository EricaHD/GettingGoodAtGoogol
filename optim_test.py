import random
from collections import defaultdict
from itertools import product

import numpy as np
from scipy import optimize

from trainer import *
from game import Game
from agent import *
from util import *


eval_games = [{'lo': 1, 'hi': 100000, 'n_idx': 50, 'replace': False, 'dist': 'uniform'},
              {'lo': 1, 'hi': 1000, 'n_idx': 50, 'replace': False, 'dist': 'uniform'},
              {'lo': 1, 'hi': 10000, 'n_idx': 50, 'replace': False, 'dist': 'uniform'},
              {'lo': 1, 'hi': 1000000, 'n_idx': 50, 'replace': False, 'dist': 'uniform'},
              {'lo': 1, 'hi': 100000, 'n_idx': 25, 'replace': False, 'dist': 'uniform'},
              {'lo': 1, 'hi': 100000, 'n_idx': 100, 'replace': False, 'dist': 'uniform'},
              {'lo': 1, 'hi': 100000, 'n_idx': 50, 'replace': True, 'dist': 'uniform'},
              {'lo': 1, 'hi': 100000, 'n_idx': 50, 'replace': False, 'dist': 'normal'}]

for i, egame in enumerate(eval_games):

    print("Game " + str(i))

    # Create game
    game_params = {'lo': egame['lo'],
                   'hi': egame['hi'],
                   'n_idx': egame['n_idx'],
                   'replace': egame['replace'],
                   'reward_fn': rewardScalar,
                   'reward': {'pos': 1, 'neg': -1},
                   'dist': egame['dist']}
    game = Game(**game_params)

    # Create agent
    agent_params = {'n_idx': egame['n_idx'],
                    'max_val': egame['hi']}
    agent = OptimalAgent(**agent_params)

    # Make agent play game
    trainer = Trainer()
    trainer.eval(game, agent, 10000, 1000, 0)  # game, agent, n_games, n_print, delay
