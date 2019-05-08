import random
from argparse import ArgumentParser
import numpy as np
from scipy.optimize import fmin

from trainer import *
from game import Game
from agent import *
from utils import *

if __name__ == '__main__':

    ap = ArgumentParser()
    
    # Agent Parameters
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
    
    # Training Game Parameters
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
    
    # Training Parameters
    ap.add_argument("-ne", "--n_episodes", type=int, default=500000,
                    help="number of Monte Carlo episodes [mc only]")
    ap.add_argument("-np", "--n_print", type=int, default=10000,
                    help="when to print [q only]")
    ap.add_argument("-d", "--delay", type=int, default=0,
                    help="time delay in training games [q only]")
    ap.add_argument("-cre", "--curr_epoch", type=int, default=1000000000,
                    help="curriculum epoch")     
    ap.add_argument("-crp", "--curr_params", type=str, default="0_0_10_-",
                    help="curriculum parameters, #_#_op when reward_fn is scalar, #_#_#_op when reward_fn is topN")    
    
    # Evaluation Game Parameters
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
    
    # Evaluation Parameters
    ap.add_argument("-nge", "--n_games_eval", type=int, default=10000,
                    help="number of evaluation games")
    ap.add_argument("-npe", "--n_print_eval", type=int, default=1000,
                    help="when to print")
    ap.add_argument("-de", "--delay_eval", type=int, default=0,
                    help="time delay in evaluation games")
    
    # Save Path
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
        reward = {'pos':int(pos), 'neg':-int(neg)}
        
        c_pos, c_neg, c_op = args['curr_params'].split('_')
        curr_params = {'pos':int(c_pos), 'neg':-int(c_neg), 'op':convertOp(c_op)}
        
    elif 'topN' in args['reward_fn']:
        reward_fn = rewardTopN
        pos, neg, n = args['reward'].split('_')
        reward = {'pos':int(pos), 'neg':-int(neg), 'n':int(n)} 
        
        c_pos, c_neg, c_n, c_op = args['curr_params'].split('_')
        curr_params = {'pos':int(c_pos), 'neg':-int(c_neg), 'n':int(c_n), 'op':convertOp(c_op)}
        
    game_params = {'lo':args['lo'],
                   'hi':args['hi'],
                   'n_idx':args['n_idx'],
                   'replace':args['replace'],
                   'reward_fn':reward_fn,
                   'reward':reward}
    
    game = Game(**game_params)
    
    ##################################################
    # SET UP Q-KEY
    ##################################################
    
    if 'bin' in args['q_key_fn']:
        i_bin, v_bin = args['q_key_params'].split('_')
        q_key_fn = qKeyMaxBin
        q_key_params = {'i_bin':int(i_bin), 'v_bin':int(v_bin)}
    elif 'binV' in args['q_key_fn']:
        i_bin, v_bin = args['q_key_params'].split('_')
        q_key_fn = qKeyMaxBinV
        q_key_params = {'i_bin':int(i_bin), 'v_bin':int(v_bin)}
    elif 'seq' in args['q_key_fn']:
        v_bin = args['q_key_params'].split('_')
        q_key_fn = qKeySeq
        q_key_params = {'v_bin':int(v_bin[0])}

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
    # F-Min
    ##################################################
    
    # def objective(_params):
    #
    #     [_g, _e] = _params
    #
    #     print("Trying params " + str(_g) + " and " + str(_e))
    #
    #     agent_params = {'gamma':_g,
    #                     'eps':_e,
    #                     'eps_decay':args['eps_decay'],
    #                     's_cost':args['s_cost'],
    #                     'q_key_fn':q_key_fn,
    #                     'q_key_params':q_key_params,
    #                     'v_fn':v_fn,
    #                     'v_key':v_key}
    #
    #     agent = MCMCAgent(**agent_params)
    #
    #     trainer_train_params = {'game':game,
    #                             'agent':agent,
    #                             'n_episodes':args['n_episodes'],
    #                             'curriculum':{'epoch':args['curr_epoch'], 'params':curr_params}}
    #
    #     trainer = MCMCTrainer()
    #
    #     percent_wins = trainer.train(**trainer_train_params)
    #     percent_losses = 1.0 - percent_wins
    #     return percent_losses
    #
    # best_params = fmin(objective, [args['gamma'], args['epsilon']], maxiter=0)  # set maxiter > 0 to use fmin
    # print("BEST GAMMA, EPSILON:", best_params)
        
    ##################################################
    # SET UP MONTE CARLO AGENT
    ##################################################
    
    agent_params = {'gamma':args['gamma'],
                    'eps':args['epsilon'],
                    'eps_decay':args['eps_decay'], 
                    's_cost':args['s_cost'],
                    'q_key_fn':q_key_fn,
                    'q_key_params':q_key_params,
                    'v_fn':v_fn,
                    'v_key':v_key}
        
    agent = MCMCAgent(**agent_params)

    trainer_train_params = {'game':game,
                            'agent':agent,
                            'n_episodes':args['n_episodes'],
                            'curriculum':{'epoch':args['curr_epoch'], 'params':curr_params}}
        
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

    eval_games = {"Eval1":{'lo':1, 'hi':100000, 'n_idx':50, 'replace':False},
                  "Eval2":{'lo':1, 'hi':1000, 'n_idx':50, 'replace':False},
                  "Eval3":{'lo':1, 'hi':10000, 'n_idx':50, 'replace':False},
                  "Eval4":{'lo':1, 'hi':1000000, 'n_idx':50, 'replace':False},
                  "Eval5":{'lo':1, 'hi':100000, 'n_idx':25, 'replace':False},
                  "Eval6":{'lo':1, 'hi':100000, 'n_idx':100, 'replace':False},
                  "Eval7":{'lo':1, 'hi':100000, 'n_idx':50, 'replace':True}}
    
    for i, game in enumerate(eval_games):
        
        ##################################################
        # TRANSFER LEARNING
        ##################################################
        
        agent = ldZipPkl(args['file_path'])
        
        game_train_params = {'lo':game['lo'],
                             'hi':game['hi'],
                             'n_idx':game['n_idx'],
                             'replace':game['replace'],
                             'reward_fn':rewardTopN,
                             'reward':{'pos':10, 'neg':-10, 'n':7}}
        
        game_train = Game(**game_train_params)
        
        trainer_train_params = {'game':game_train,
                                'agent':agent,
                                'n_episodes':10000,
                                'curriculum':{'epoch':1000000000, 'params':{}}}
        
        trainer.train(**trainer_train_params)
        
        ##################################################
        # EVALUATION
        ##################################################
        
        game_eval_params = {'lo':game['lo'],
                            'hi':game['hi'],
                            'n_idx':game['n_idx'],
                            'replace':game['replace'],
                            'reward_fn':rewardScalar,
                            'reward':{'pos':1, 'neg':-1}}
            
        game_eval = Game(**game_eval_params)
        
        trainer_eval_params = {'game':game_eval,
                               'agent':agent,
                               'n_episodes':10000,
                               'curriculum':{'epoch':1000000000, 'params':{}}}
         
        if i == 0:
            _, stop_choices = train.eval(**trainer_eval_params)
            svZipPkl(stop_choices, args['sc_file_path'])
        else:
            train.eval(**trainer_eval_params)
