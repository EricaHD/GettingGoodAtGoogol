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
    ap.add_argument("-al", "--alpha", type=float, default=0.01,
                    help="learning rate [q only]")
    ap.add_argument("-ald", "--alpha_decay", type=float, default=0.00001,
                    help="learning rate decay factor [q only]")
    ap.add_argument("-as", "--alpha_step", type=int, default=10000,
                    help="learning rate decays every alpha_step turns [q only]")
    ap.add_argument("-g", "--gamma", type=float, default=0.9,
                    help="discount factor")
    ap.add_argument("-e", "--epsilon", type=float, default=0.1,
                    help="the probability of exploration")
    ap.add_argument("-ed", "--eps_decay", type=float, default=0.00001,
                    help="epsilon decay factor")
    ap.add_argument("-s", "--s_cost", type=float, default=0,
                    help="search cost")
    ap.add_argument("-ql", "--q_learn", type=bool, default=False,
                    help="SARSA when True, Q-learning when False [q only]")
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
    ap.add_argument("-ng", "--n_games", type=int, default=500000,
                    help="number of training games [q only]")
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
    # SET UP Q-Key
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
    
    def objective(_params):
        
        [_a, _g, _e] = _params
        
        print("Trying params " + str(_a) + " and " + str(_g) + " and " + str(_e))
        
        agent_params = {'alpha':_a,
                        'alpha_decay':args['alpha_decay'],
                        'alpha_step':args['alpha_step'],
                        'gamma':_g,
                        'eps':_e, 
                        'eps_decay':args['eps_decay'], 
                        's_cost':args['s_cost'],
                        'sarsa':args['q_learn'],
                        'q_key_fn':q_key_fn,
                        'q_key_params':q_key_params,
                        'v_fn':v_fn,
                        'v_key':v_key}
        
        agent = QAgent(**agent_params)
        
        trainer_train_params = {'game':game,
                                'agent':agent,
                                'n_games':args['n_games'],
                                'n_print':args['n_print'],
                                'delay':args['delay'],
                                'curriculum':{'epoch':args['curr_epoch'], 'params':curr_params}}
        
        trainer = QTrainer()
        
        percent_wins = trainer.train(**trainer_train_params)
        percent_losses = 1.0 - percent_wins
        return percent_losses
    
    best_params = fmin(objective, [args['alpha'], args['gamma'], args['epsilon']], maxiter=0)  # set maxiter > 0 to use fmin
    print("BEST ALPHA, GAMMA, EPSILON:", best_params)
    
    ##################################################
    # SET UP Q-LEARNING AGENT
    ##################################################   
    
    agent_params = {'alpha':best_params[0],
                    'alpha_decay':args['alpha_decay'],
                    'alpha_step':args['alpha_step'],
                    'gamma':best_params[1],
                    'eps':best_params[2], 
                    'eps_decay':args['eps_decay'], 
                    's_cost':args['s_cost'],
                    'sarsa':args['q_learn'],
                    'q_key_fn':q_key_fn,
                    'q_key_params':q_key_params,
                    'v_fn':v_fn,
                    'v_key':v_key}
        
    agent = QAgent(**agent_params)
        
    trainer_train_params = {'game':game,
                            'agent':agent,
                            'n_games':args['n_games'],
                            'n_print':args['n_print'],
                            'delay':args['delay'],
                            'curriculum':{'epoch':args['curr_epoch'], 'params':curr_params}}
        
    trainer = QTrainer()
        
    ##################################################
    # TRAINING
    ##################################################
    
    print('TRAINING')
    trainer.train(**trainer_train_params)
    print('*' * 89)
    print('*' * 89)
    
    ##################################################
    # SET UP EVALUATION
    ##################################################
    
    if 'scalar' in args['reward_fn_eval']:
        reward_fn_eval = rewardScalar
        pos, neg = args['reward_eval'].split('_')
        reward_eval = {'pos':int(pos), 'neg':-int(neg)}
    elif 'topN' in args['reward_fn_eval']:
        reward_fn_eval = rewardTopN
        pos, neg, n = args['reward_eval'].split('_')
        reward_eval = {'pos':int(pos), 'neg':-int(neg), 'n':int(n)} 
        
        
    game_eval_params = {'lo':args['lo_eval'],
                        'hi':args['hi_eval'],
                        'n_idx':args['n_idx_eval'],
                        'replace':args['replace_eval'],
                        'reward_fn':reward_fn_eval,
                        'reward':reward_eval}
    
    game_eval = Game(**game_eval_params)
    
    trainer_eval_params = {'game':game_eval,
                           'agent':agent,
                           'n_games':args['n_games_eval'],
                           'n_print':args['n_print_eval'],
                           'delay':args['delay_eval']}
    
    ##################################################
    # EVALUATION
    ##################################################
    
    print('EVAL')
    trainer.eval(**trainer_eval_params)
    print('*' * 89)
    print('*' * 89)
    
    ##################################################
    # SAVE
    ##################################################
    
    svZipPkl(agent, args['file_path'])
    print("AGENT STORED AT: {}".format(args['file_path']))
