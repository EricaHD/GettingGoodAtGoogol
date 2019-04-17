import random
from argparse import ArgumentParser

import numpy as np

from trainer import *
from game import Game
from agent import *
from utils import *

if __name__ == '__main__':

    ap = ArgumentParser()
    
    #Agent
    ap.add_argument("-a", "--agent", default='q_learn',
                    help="Agent Type")
    ap.add_argument("-ap", "--agent_path",
                    help="Agent Path")
    
    #Game Parameters
    ap.add_argument("-lo", "--lo", default=1,
                    help="Lo")
    ap.add_argument("-hi", "--hi", default=100,
                    help="Hi")
    ap.add_argument("-ni", "--n_idx", default=25,
                    help="N-idx")
    ap.add_argument("-rp", "--replace", default=False,
                    help="Replacement")
    ap.add_argument("-r", "--reward", default="scalar_1_1",
                    help="Reward Fn")
    
    #Eval Parameters
    ap.add_argument("-ng", "--n_games", default=1000000,
                    help="N-Games")
    ap.add_argument("-np", "--n_print", default=100000,
                    help="N-Print")
    ap.add_argument("-d", "--delay", default=0,
                    help="Time Delay")
    
    
    args = vars(ap.parse_args())
    
    path_params = ldZipPkl(args['agent_path'])
    
    ###SET UP GAME
    if "scalar" in args['reward']:
        pos_reward, neg_reward = args['reward'].split("_")[1:]
        reward_fn = lambda g: rewardScalar(g, int(pos_reward), -int(neg_reward)) 
    elif 'topN' in args['reward']:
        pos_reward, neg_reward, n = args['reward'].split("_")[1:]
        n_pct = int(int(n)/100 * int(args['n_idx']))
        reward_fn = lambda g: rewardTopN(g, int(pos_reward), -int(neg_reward), n_pct) 
        
    game_params = {'lo':int(args['lo']),
                   'hi':int(args['hi']),
                   'n_idx':int(args['n_idx']),
                   'replace':bool(args['replace']),
                   'reward_fn':reward_fn,
                  }
    
    
    game = Game(**game_params)
    
    ###SET UP AGENT
    if args['agent'] == "q_learn":
        agent = QAgent(**path_params['agent_params'])
        agent.Q = path_params['agent_Q']
        trainer = QTrainer()
    elif args['agent'] == 'mcmc':
        agent = MCMCAgent(**path_params['agent_params'])
        agent.Q = path_params['agent_Q']
        agent.policy = path_params['agent_policy']
        trainer = MCMCTrainer()
    
        

    trainer_params = {'game':game,
                      'agent':agent,
                      'n_games':int(args['n_games']),
                      'n_print':int(args['n_print']),
                      'delay':int(args['delay'])}

    trainer.eval(**trainer_params)
    print("*" * 89)
    print("*" * 89)
    
    
        
    
