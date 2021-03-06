import random
from argparse import ArgumentParser
import numpy as np
from scipy.optimize import fmin  # can use fmin to tune hyperparameters

from trainer import *
from game import Game
from agent import *
from util import *

if __name__ == '__main__':

    ap = ArgumentParser()
    
    # Agent parameters
    ap.add_argument("-g", "--gamma", type=float, default=0.9,
                    help="discount factor")
    ap.add_argument("-e", "--epsilon", type=float, default=0.1,
                    help="the probability of exploration")
    ap.add_argument("-ed", "--eps_decay", type=float, default=0.00001,
                    help="epsilon decay factor")
    ap.add_argument("-s", "--s_cost", type=float, default=0,
                    help="search cost")
    ap.add_argument("-qkf", "--q_key_fn", type=str, default="bin",
                    help="can be bin or seq")
    ap.add_argument("-qkp", "--q_key_params", type=str, default="2_20",
                    help="# when q_key_fn is seq, #_# when q_key_fn is bin")
    ap.add_argument("-vf", "--v_fn", type=str, default="vIdx",
                    help="can be vMax or vSeq or vIdx")
    
    # Training game parameters
    ap.add_argument("-lo", "--lo", type=int, default=1,
                    help="lowest value possible in training games")
    ap.add_argument("-hi", "--hi", type=int, default=100000,
                    help="highest value possible in training games")
    ap.add_argument("-ni", "--n_idx", type=int, default=50,
                    help="number of cards in training games")
    ap.add_argument("-rp", "--replace", type=bool, default=False,
                    help="numbers in training games can repeat when True, numbers are distinct when False")
    ap.add_argument("-r", "--reward_fn", type=str, default="topN",
                    help="reward function in training games, can be scalar or topN")
    ap.add_argument("-rps", "--reward", type=str, default="10_10_7",
                    help="#_# when reward_fn is scalar, #_#_# when reward_fn is topN")
    
    # Training parameters
    ap.add_argument("-ne", "--n_games", type=int, default=500000,
                    help="number of Monte Carlo episodes")
    ap.add_argument("-np", "--n_print", type=int, default=10000,
                    help="when to print [q only]")
    ap.add_argument("-d", "--delay", type=int, default=0,
                    help="time delay in training games [q only]")   
    
    # Evaluation game parameters
    ap.add_argument("-loe", "--lo_eval", type=int, default=1,
                    help="lowest value possible in evaluation games")
    ap.add_argument("-hie", "--hi_eval", type=int, default=100000,
                    help="highest value possible in evaluation games")
    ap.add_argument("-nie", "--n_idx_eval", type=int, default=50,
                    help="number of cards in training games")
    ap.add_argument("-rpe", "--replace_eval", type=bool, default=False,
                    help="numbers in evaluation games can repeat when True, numbers are distinct when False")
    ap.add_argument("-re", "--reward_fn_eval", type=str, default="scalar",
                    help="reward function in evaluation games, can be scalar or topN")
    ap.add_argument("-rpse", "--reward_eval", type=str, default="1_1",
                    help="#_# when reward_fn_eval is scalar, #_#_# when reward_fn_eval is topN")
    
    # Evaluation parameters
    ap.add_argument("-nge", "--n_games_eval", type=int, default=10000,
                    help="number of evaluation games")
    ap.add_argument("-npe", "--n_print_eval", type=int, default=1000,
                    help="when to print")
    ap.add_argument("-de", "--delay_eval", type=int, default=0,
                    help="time delay in evaluation games")
    
    # Save path
    ap.add_argument("-fp", "--file_path",
                    help="file path used for saving")
    ap.add_argument("-scfp", "--sc_file_path",
                    help="file path used for saving stopping choices")
    
    args = vars(ap.parse_args())
    
    ##################################################
    # SET UP GAME
    ##################################################
    
    if 'scalar' in args['reward_fn']:
        reward_fn = rewardScalar
        pos, neg = args['reward'].split('_')
        reward = {'pos': int(pos), 'neg': -int(neg)}
        
    elif 'topN' in args['reward_fn']:
        reward_fn = rewardTopN
        pos, neg, n = args['reward'].split('_')
        reward = {'pos': int(pos), 'neg': -int(neg), 'n': int(n)}
        
    game_params = {'lo': args['lo'],
                   'hi': args['hi'],
                   'n_idx': args['n_idx'],
                   'replace': args['replace'],
                   'reward_fn': reward_fn,
                   'reward': reward,
                   'dist': 'uniform'}
    
    game = Game(**game_params)
    
    ##################################################
    # SET UP Q-KEY
    ##################################################
    
    if 'bin' in args['q_key_fn']:
        i_bin, v_bin = args['q_key_params'].split('_')
        q_key_fn = qKeyMaxBin
        q_key_params = {'i_bin': int(i_bin), 'v_bin': int(v_bin)}
    elif 'binV' in args['q_key_fn']:
        i_bin, v_bin = args['q_key_params'].split('_')
        q_key_fn = qKeyMaxBinV
        q_key_params = {'i_bin': int(i_bin), 'v_bin': int(v_bin)}
    elif 'seq' in args['q_key_fn']:
        v_bin = args['q_key_params'].split('_')
        q_key_fn = qKeySeq
        q_key_params = {'v_bin': int(v_bin[0])}

    if args['v_fn'] == 'vMax':
        v_fn = vMax
        v_key = -1
    elif args['v_fn'] == 'vSeq':
        v_fn = vSeq
        v_key = str([0])
    elif args['v_fn'] == 'vIdx':
        v_fn = vIdx
        v_key = 0

    ##################################################
    # SET UP MONTE CARLO AGENT
    ##################################################
    
    agent_params = {'gamma': args['gamma'],
                    'eps': args['epsilon'],
                    'eps_decay': args['eps_decay'],
                    's_cost': args['s_cost'],
                    'q_key_fn': q_key_fn,
                    'q_key_params': q_key_params,
                    'v_fn': v_fn,
                    'v_key': v_key}
        
    agent = MCMCAgent(**agent_params)

    trainer_train_params = {'game': game,
                            'agent': agent,
                            'n_games': args['n_games']}
        
    trainer = MCMCTrainer()
        
    ##################################################
    # TRAINING
    ##################################################
    
    print('TRAINING')
    trainer.train(**trainer_train_params)
    print('*' * 89)
    print('*' * 89)
    
    svZipPkl(agent, args['file_path'])
    
    ##################################################
    # TRANSFERING LEARNING & EVALUATION
    ##################################################

    eval_games = [{'lo': 1, 'hi': 100000, 'n_idx': 50, 'replace': False, 'dist': 'uniform'},
                  {'lo': 1, 'hi': 1000, 'n_idx': 50, 'replace': False, 'dist': 'uniform'},
                  {'lo': 1, 'hi': 10000, 'n_idx': 50, 'replace': False, 'dist': 'uniform'},
                  {'lo': 1, 'hi': 1000000, 'n_idx': 50, 'replace': False, 'dist': 'uniform'},
                  {'lo': 1, 'hi': 100000, 'n_idx': 25, 'replace': False, 'dist': 'uniform'},
                  {'lo': 1, 'hi': 100000, 'n_idx': 100, 'replace': False, 'dist': 'uniform'},
                  {'lo': 1, 'hi': 100000, 'n_idx': 50, 'replace': True, 'dist': 'uniform'},
                  {'lo': 1, 'hi': 100000, 'n_idx': 50, 'replace': False, 'dist': 'normal'}]
    
    for i, game in enumerate(eval_games):
        
        ##################################################
        # TRANSFER LEARNING
        ##################################################
        
        agent = ldZipPkl(args['file_path'])
        
        game_train_params = {'lo': game['lo'],
                             'hi': game['hi'],
                             'n_idx': game['n_idx'],
                             'replace': game['replace'],
                             'reward_fn': rewardTopN,
                             'reward': {'pos': 10, 'neg': -10, 'n': 7},
                             'dist': game['dist']}
        
        game_train = Game(**game_train_params)
        
        trainer_train_params = {'game': game_train,
                                'agent': agent,
                                'n_games': 10000}
        
        trainer.train(**trainer_train_params)
        
        ##################################################
        # EVALUATION
        ##################################################
        
        game_eval_params = {'lo': game['lo'],
                            'hi': game['hi'],
                            'n_idx': game['n_idx'],
                            'replace': game['replace'],
                            'reward_fn': rewardScalar,
                            'reward': {'pos': 1, 'neg': -1},
                            'dist': game['dist']}
            
        game_eval = Game(**game_eval_params)
        
        trainer_eval_params = {'game': game_eval,
                               'agent': agent,
                               'n_games': 10000,
                               'n_print': 1000,
                               'delay': 0}
        
        if i == 0:
            _, stop_choices = trainer.eval(**trainer_eval_params)
            svZipPkl(stop_choices, args['sc_file_path'])
        else:
            trainer.eval(**trainer_eval_params)
