import random
from argparse import ArgumentParser

import numpy as np

from env import Env
from game import Game
from agent import QAgent
from utils import *

if __name__ == '__main__':

    ap = ArgumentParser()
    
    #Agent
    ap.add_argument("-a", "--agent", default='q_learn',
                    help="Agent Type")
    
    #Game Parameters
    ap.add_argument("-lo", "--lo", default=1,
                    help="Lo")
    ap.add_argument("-hi", "--hi", default=100,
                    help="Hi")
    ap.add_argument("-ns", "--n_states", default=25,
                    help="N-States")
    ap.add_argument("-rp", "--replace", default=False,
                    help="SARSA")
    
    #Agent Parameters
    ap.add_argument("-al", "--alpha", default=0.01,
                    help="Alpha")
    ap.add_argument("-ald", "--alpha_decay", default=1e-5,
                    help="Alpha Decay")
    ap.add_argument("-as", "--alpha_step", default=1000,
                    help="Alpha Step")
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
    
    #Training Parameters
    ap.add_argument("-ng", "--n_games", default=1000000,
                    help="N-Games")
    ap.add_argument("-r", "--reward", default="scalar_10_1",
                    help="Reward Fn")
    ap.add_argument("-np", "--n_print", default=100000,
                    help="N-Print")
    ap.add_argument("-d", "--delay", default=0,
                    help="Time Delay")
    
    
    #Eval-Game Parameters
    ap.add_argument("-loe", "--lo_eval", default=1,
                    help="Lo")
    ap.add_argument("-hie", "--hi_eval", default=100,
                    help="Hi")
    ap.add_argument("-nse", "--n_states_eval", default=25,
                    help="N-States")
    ap.add_argument("-rpe", "--replace_eval", default=False,
                    help="SARSA")
    
    #Eval Parameters
    ap.add_argument("-nge", "--n_games_eval", default=10000,
                    help="N-Games Eval")
    ap.add_argument("-re", "--reward_eval", default="scalar_10_1",
                    help="Reward Fn Eval")
    ap.add_argument("-npe", "--n_print_eval", default=1000,
                    help="N-Print")
    ap.add_argument("-de", "--delay_eval", default=0,
                    help="Time Delay")
    
    #Save Path
    ap.add_argument("-fp", "--file_path",
                    help="Save File Path")
    
    
    args = vars(ap.parse_args())
    
    print("INITIALIZING GAME")
    ###SET UP GAME
    game_params = {'lo':int(args['lo']),
                   'hi':int(args['hi']),
                   'n_states':int(args['n_states']),
                   'replace':bool(args['replace'])}
    
    
    game = Game(**game_params)
    
    print("*"*89)
    print("*"*89)
    
    print("INITIALIZING AGENT")
    ###SET UP AGENT
    if args['agent'] == "q_learn":
        if "bin" in args['q_key_fn']:
            s_bin, v_bin = args['q_key_fn'].split("_")[1:]
            q_key_fn = lambda p, q: qKeyMaxBin(p, q, int(s_bin), int(v_bin))
        elif "seq" in args['q_key_fn']:
            s_bin = args['q_key_fn'].split("_")[1]
            q_key_fn = lambda p, q: qKeySeqBin(p, q, int(s_bin))
    
        agent_params = {'alpha':float(args['alpha']),
                        'alpha_decay':float(args['alpha_decay']),
                        'alpha_step':int(args['alpha_step']),
                      	'gamma':float(args['gamma']),
                      	'eps':float(args['epsilon']), 
                      	'eps_decay':float(args['eps_decay']), 
                      	's_cost':float(args['s_cost']),
                      	'sarsa':bool(args['q_learn']),
                      	'q_key_fn':q_key_fn,
                      	'q_key':args['q_key']}
        
        agent = QAgent(**agent_params)
        
    print("*"*89)
    print("*"*89)
    
    print("INITIALIZING TRAINING")
    ###TRAIN
    if "scalar" in args['reward']:
        pos_reward, neg_reward = args['reward'].split("_")[1:]
        reward_fn = lambda g, gp: rewardScalar(g, gp, int(pos_reward), -int(neg_reward)) 
    elif 'topN' in args['reward']:
        pos_reward, neg_reward, n = args['reward'].split("_")[1:]
        reward_fn = lambda g, gp: rewardTopN(g, gp, int(pos_reward), -int(neg_reward), int(n)) 
        
    env_train_params = {'game':game,
                  'agent':agent,
                  'n_games':int(args['n_games']),
                  'reward_fn': reward_fn,
                  'n_print':int(args['n_print']),
                  'delay':int(args['delay'])}
    
    env = Env()
    env.train(**env_train_params)
    print("*" * 89)
    print("*" * 89)
    
    
    print("INITIALIZING EVAL")
    ###EVAL
    game_eval_params = {'lo':int(args['lo_eval']),
                       'hi':int(args['hi_eval']),
                       'n_states':int(args['n_states_eval']),
                       'replace':bool(args['replace_eval'])}
    
    
    game_eval = Game(**game_eval_params)
    
    if "scalar" in args['reward_eval']:
        pos_reward, neg_reward = args['reward_eval'].split("_")[1:]
        reward_fn_eval = lambda g, gp: rewardScalar(g, gp, int(pos_reward), -int(neg_reward)) 
    elif 'topN' in args['reward_eval']:
        pos_reward, neg_reward, n = args['reward_eval'].split("_")[1:]
        reward_fn_eval = lambda g, gp: rewardTopN(g, gp, int(pos_reward), -int(neg_reward), int(n)) 
    
    env_eval_params = {'game':game_eval,
                       'agent':agent,
                       'n_games':int(args['n_games_eval']),
                       'reward_fn': reward_fn_eval,
                       'n_print':int(args['n_print']),
                       'delay':int(args['delay'])}
    
    env.eval(**env_eval_params)
    print("*" * 89)
    print("*" * 89)
    
    ###SAVE
    if args['agent'] == "q_learn":
        save_params = {"game_params":game_params,
                       "game_eval_params":game_eval_params,
                       "agent_params":agent_params,
                       "agent_Q":agent.Q}
        svZipPkl(save_params, args['file_path'])
        
        print("Q-VALUES STORED AT: {}".format(args['file_path']))
    
    
        
    
