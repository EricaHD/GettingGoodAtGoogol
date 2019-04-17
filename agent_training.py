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
    
    #Game Parameters
    ap.add_argument("-lo", "--lo", default=1,
                    help="Lo")
    ap.add_argument("-hi", "--hi", default=100,
                    help="Hi")
    ap.add_argument("-ni", "--n_idx", default=25,
                    help="N-Idx")
    ap.add_argument("-rp", "--replace", default=False,
                    help="Replacement")
    ap.add_argument("-r", "--reward", default="scalar_10_1",
                    help="Reward Fn")
    
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
    ap.add_argument("-vf", "--v_fn", default="vMax",
                    help="Val Fn")
    ap.add_argument("-pkf", "--p_key_fn", default="bin_1_2",
                    help="Policy Fn")
    
    #Training Parameters
    ap.add_argument("-ng", "--n_games", default=1000000,
                    help="N-Games")
    ap.add_argument("-ne", "--n_episodes", default=1000000,
                    help="N-Episodes")
    ap.add_argument("-np", "--n_print", default=100000,
                    help="N-Print")
    ap.add_argument("-d", "--delay", default=0,
                    help="Time Delay")
    
    
    #Eval-Game Parameters
    ap.add_argument("-loe", "--lo_eval", default=1,
                    help="Lo")
    ap.add_argument("-hie", "--hi_eval", default=100,
                    help="Hi")
    ap.add_argument("-nie", "--n_idx_eval", default=25,
                    help="N-Idx")
    ap.add_argument("-rpe", "--replace_eval", default=False,
                    help="Replacement")
    ap.add_argument("-re", "--reward_eval", default="scalar_10_1",
                    help="Reward Fn")
    
    #Eval Parameters
    ap.add_argument("-nge", "--n_games_eval", default=10000,
                    help="N-Games Eval")
    ap.add_argument("-npe", "--n_print_eval", default=1000,
                    help="N-Print")
    ap.add_argument("-de", "--delay_eval", default=0,
                    help="Time Delay")
    
    #Save Path
    ap.add_argument("-fp", "--file_path",
                    help="Save File Path")
    
    
    args = vars(ap.parse_args())
    
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
                   'reward_fn':reward_fn}
    
    game = Game(**game_params)
    
    ###SET UP AGENT
    if args['agent'] == "q_learn":
        if "bin" in args['q_key_fn']:
            i_bin, v_bin = args['q_key_fn'].split("_")[1:]
            q_key_fn = lambda p, i, v: qKeyMaxBin(p, i, v, int(i_bin), int(v_bin))
            
        if args['v_fn'] == "vMax":
            v_fn = vMax
    
        agent_params = {'alpha':float(args['alpha']),
                        'alpha_decay':float(args['alpha_decay']),
                        'alpha_step':int(args['alpha_step']),
                      	'gamma':float(args['gamma']),
                      	'eps':float(args['epsilon']), 
                      	'eps_decay':float(args['eps_decay']), 
                      	's_cost':float(args['s_cost']),
                      	'sarsa':bool(args['q_learn']),
                      	'q_key_fn':q_key_fn,
                      	'v_fn':v_fn}
        
        agent = QAgent(**agent_params)
        
        trainer_train_params = {'game':game,
                                'agent':agent,
                                'n_games':int(args['n_games']),
                                'n_print':int(args['n_print']),
                                'delay':int(args['delay'])}
        
        trainer = QTrainer()
        
    elif args['agent'] == "mcmc":
        if "bin" in args['q_key_fn']:
            i_bin, v_bin = args['q_key_fn'].split("_")[1:]
            q_key_fn = lambda p, i, v: qKeyMaxBin(p, i, v, int(i_bin), int(v_bin))
        if "bin" in args['p_key_fn']:
            i_bin, v_bin = args['p_key_fn'].split("_")[1:]
            p_key_fn = lambda p, i, v: qKeyMaxBin(p, i, v, int(i_bin), int(v_bin))
            
        if args['v_fn'] == "vMax":
            v_fn = vMax
        
        agent_params = {'gamma':float(args['gamma']),
                      	'eps':float(args['epsilon']), 
                      	'eps_decay':float(args['eps_decay']), 
                      	's_cost':float(args['s_cost']),
                      	'q_key_fn':q_key_fn,
                        'p_key_fn':p_key_fn,
                      	'v_fn':v_fn}
        
        agent = MCMCAgent(**agent_params)
        
        trainer_train_params = {'game':game,
                                'agent':agent,
                                'n_episodes':int(args['n_episodes'])}
        
        trainer = MCMCTrainer()
    
    print("TRAINING")
    trainer.train(**trainer_train_params)
    print("*" * 89)
    print("*" * 89)
    
    
    if "scalar" in args['reward_eval']:
        pos_reward, neg_reward = args['reward_eval'].split("_")[1:]
        reward_fn = lambda g: rewardScalar(g, int(pos_reward), -int(neg_reward)) 
    elif "topN" in args['reward_eval']:    
        pos_reward, neg_reward, n = args['reward_eval'].split("_")[1:]
        n_pct = int(int(n)/100 * int(args['n_idx_eval']))
        reward_fn = lambda g: rewardTopN(g, tint(pos_reward), -int(neg_reward), n_pct) 
        
    game_eval_params = {'lo':int(args['lo_eval']),
                   'hi':int(args['hi_eval']),
                   'n_idx':int(args['n_idx_eval']),
                   'replace':bool(args['replace_eval']),
                   'reward_fn':reward_fn}
    
    game_eval = Game(**game_eval_params)
    
    trainer_eval_params = {'game':game_eval,
                           'agent':agent,
                           'n_games':int(args['n_games_eval']),
                           'n_print':int(args['n_print']),
                           'delay':int(args['delay'])}
    
    print("EVAL")
    trainer.eval(**trainer_eval_params)
    print("*" * 89)
    print("*" * 89)
    
    ###SAVE
    if args['agent'] == "q_learn":
        save_params = {"agent_params":agent_params,
                       "agent_Q":agent.Q}
        svZipPkl(save_params, args['file_path'])
        print("AGENT STORED AT: {}".format(args['file_path']))
    elif args['agent'] == 'mcmc':
        save_params = {"agent_params":agent_params,
                       "agent_Q":agent.Q,
                       "agent_policy":agent.policy}
        svZipPkl(save_params, args['file_path'])
        print("AGENT STORED AT: {}".format(args['file_path']))
        
        
    
    
        
    
