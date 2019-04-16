import random
from argparse import ArgumentParser

import numpy as np

from env import Env
from game import Game
from agent import QAgent
from utils import qKeyMaxBin, qKeySeqBin, rewardScalar, rewardTopN

import pickle as pkl

if __name__ == '__main__':

    ap = ArgumentParser()
    
    #Agent
    ap.add_argument("-a", "--agent", default='q_learn',
                    help="Agent Type")
    
    #Game Parameters
    ap.add_argument("-lo", "--lo", default=1,
                    help="Lo")
    ap.add_argument("-hi", "--hi", default=1000,
                    help="Hi")
    ap.add_argument("-ns", "--n_states", default=50,
                    help="N-States")
    ap.add_argument("-rp", "--replace", default=False,
                    help="SARSA")
    
    #Training Parameters
    ap.add_argument("-ng", "--n_games", default=10_000,
                    help="N-Games")
    ap.add_argument("-r", "--reward", default="scalar_10_1",
                    help="Reward Fn")
    
    #Agent Parameters
    ap.add_argument("-al", "--alpha", default=0.001,
                    help="Learning Rate")
    ap.add_argument("-g", "--gamma", default=0.9,
                    help="Reward Discount Rate")
    ap.add_argument("-e", "--epsilon", default=0.1,
                    help="Epsilon")
    ap.add_argument("-ed", "--eps_decay", default=1e-5,
                    help="Epsilon Decay")
    ap.add_argument("-s", "--s_cost", default=0,
                    help="Search Cost")
    ap.add_argument("-ql", "--q_learn", default=False,
                    help="SARSA")
    ap.add_argument("-qkf", "--q_key_fn", default="bin_1_2",
                    help="Q-Key Fn")
    ap.add_argument("-qk", "--q_key", default="0_0",
                    help="Q-Key")
    
    #Save Path
    ap.add_argument("-fp", "--file_path",
                    help="Save File Path")
    
    
    args = vars(ap.parse_args())
    
    ###SET UP GAME
    game_params = {'lo':int(args['lo']),
                   'hi':int(args['hi']),
                   'n_states':int(args['n_states']),
                   'replace':bool(args['replace'])}
    
    game = Game(**game_params)
    
    ###SET UP AGENT
    if args['agent'] == "q_learn":
        if "bin" in args['q_key_fn']:
            s_bin, v_bin = args['q_key_fn'].split("_")[1:]
            q_key_fn = lambda p, q: qKeyMaxBin(p, q, int(s_bin), int(v_bin))
        elif "seq" in args['q_key_fn']:
            s_bin = args['q_key_fn'].split("_")[1]
            q_key_fn = lambda p, q: qKeySeqBin(p, q, int(s_bin))
    
        agent_params = {'alpha':float(args['alpha']),
                      	'gamma':float(args['gamma']),
                      	'eps':float(args['epsilon']), 
                      	'eps_decay':float(args['eps_decay']), 
                      	's_cost':float(args['s_cost']),
                      	'sarsa':bool(args['q_learn']),
                      	'q_key_fn':q_key_fn,
                      	'q_key':args['q_key']}
        
        agent = QAgent(**agent_params)
    
    ###SET UP ENV
    if "scalar" in args['reward']:
        pos_reward, neg_reward = args['reward'].split("_")[1:]
        reward_fn = lambda g, gp: rewardScalar(g, gp, int(pos_reward), -int(neg_reward)) 
    elif 'topN' in args['reward']:
        pos_reward, neg_reward, n = args['reward'].split("_")[1:]
        reward_fn = lambda g, gp: rewardTopN(g, gp, int(pos_reward), -int(neg_reward), int(n)) 
        
    env_params = {'game':game,
                  'agent':agent,
                  'n_games':int(args['n_games']),
                  'reward_fn': reward_fn,
                  'verbose':True}
    
    env = Env()
    
    env.train(**env_params)
        
    with open(args['file_path'], 'wb') as file:
        pkl.dump(dict(agent.Q), file)
        
    print("Agent Saved at: {}".format(args['file_path']))
