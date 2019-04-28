import random
from argparse import ArgumentParser
import numpy as np
from scipy.optimize import fmin

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

torch.manual_seed(3001)

from trainer import *
from game import Game
from agent import *
from utils import *
from networks import *

if __name__ == '__main__':

    ap = ArgumentParser()
    
    ap.add_argument("-dev", "--device", type=str, default="cpu",
                    help="Cuda Device")
    
    
    # Agent Parameters
    ap.add_argument("-net", "--net", type=str, default="basic",
                    help="Network")
    ap.add_argument("-netp", "--net_params", type=str, default="2_256",
                    help="Network Params")
    ap.add_argument("-b", "--batch_size", type=int, default=128,
                    help="Batch Size")
    ap.add_argument("-g", "--gamma", type=float, default=0.9,
                    help="discount factor")
    ap.add_argument("-e", "--epsilon", type=float, default=0.1,
                    help="the probability of exploration")
    ap.add_argument("-ed", "--eps_decay", type=float, default=1e-5,
                    help="epsilon decay factor")
    ap.add_argument("-s", "--s_cost", type=float, default=0,
                    help="search cost")
    ap.add_argument("-tu", "--target_update", type=int, default=1000,
                    help="Target Net Update")
    ap.add_argument("-o", "--optimizer", type=str, default="adam",
                    help="Optimizer")
    ap.add_argument("-loss", "--loss", type=str, default="huber",
                    help="Loss Fn")
    ap.add_argument("-mem", "--mem_size", type=int, default=100000,
                    help="Memory Size")
    ap.add_argument("-pts", "--p_to_s", type=str, default="stateMax",
                    help="Params to State fn")
    ap.add_argument("-vf", "--v_fn", type=str, default="vMax",
                    help="V Fn")
    
    # Training Game Parameters
    ap.add_argument("-lo", "--lo", type=int, default=1,
                    help="lowest value possible in training games")
    ap.add_argument("-hi", "--hi", type=int, default=100,
                    help="highest value possible in training games")
    ap.add_argument("-ni", "--n_idx", type=int, default=25,
                    help="number of cards in training games")
    ap.add_argument("-rp", "--replace", type=bool, default=False,
                    help="numbers in training games can repeat when True, numbers are distinct when False")
    ap.add_argument("-r", "--reward_fn", type=str, default="topN",
                    help="reward function in training games, can be scalar or topN")
    ap.add_argument("-rps", "--reward", type=str, default="5_5_5",
                    help="#_# when reward_fn is scalar, #_#_# when reward_fn is topN")
    
    # Training Parameters
    ap.add_argument("-ng", "--n_games", type=int, default=1000000,
                    help="number of training games [q only]")
    ap.add_argument("-ne", "--n_episodes", type=int, default=1000000,
                    help="number of Monte Carlo episodes [mc only]")
    ap.add_argument("-np", "--n_print", type=int, default=100000,
                    help="when to print [q only]")
    ap.add_argument("-d", "--delay", type=int, default=0,
                    help="time delay in training games [q only]")
    ap.add_argument("-cre", "--curr_epoch", type=int, default=100000,
                    help="curriculum epoch")     
    ap.add_argument("-crp", "--curr_params", type=str, default="0_0_10_-",
                    help="curriculum parameters, #_#_op when reward_fn is scalar, #_#_#_op when reward_fn is topN")    
    
    # Evaluation Game Parameters
    ap.add_argument("-loe", "--lo_eval", type=int, default=1,
                    help="lowest value possible in evaluation games")
    ap.add_argument("-hie", "--hi_eval", type=int, default=100,
                    help="highest value possible in evaluation games")
    ap.add_argument("-nie", "--n_idx_eval", type=int, default=25,
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
    # SET UP AGENT
    ##################################################
    
    device = torch.device(args['device'])
    
    if args['net'] == "basic":
        inp_size, hid_size = args['net_params'].split('_')
        
        net_params = {'inp_size':int(inp_size),
                      'hid_size':int(hid_size),
                      'out_size':2
                     }

        policy_net = BasicDQN(**net_params).to(device)
        target_net = BasicDQN(**net_params).to(device)
        
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    
    if args['p_to_s'] == "stateMax":
        p_to_s = stateMax
    elif args['p_to_s'] == "stateMaxV":
        p_to_s = stateMaxV
        
    if args['optimizer'] == "adam":
        optimizer = optim.Adam(policy_net.parameters())
    elif args['optimizer'] == "rmsprop":
        optimizer = optim.RMSprop(policy_net.parameters())
    
    
    if args['loss'] == "mse":
        loss = nn.MSELoss()
    elif args['loss'] == "mae":
        loss = nn.L1Loss()
    elif args['loss'] == "huber":
        loss = nn.SmoothL1Loss()

    memory = ReplayMemory(args['mem_size']) 
    
    if args['v_fn'] == 'vMax':
        v_fn = vMax
        v_key = -1
    elif args['v_fn'] == 'vIdx':
        v_fn = vIdx
        v_key = 0
    
    dqa_params = {'batch_size':args['batch_size'],
                  'gamma':args['gamma'],
                  'eps':args['epsilon'],
                  'eps_decay':args['eps_decay'],
                  'target_update':args['target_update'],
                  'p_to_s':p_to_s,
                  'p_net':policy_net,
                  't_net':target_net,
                  'optimizer':optimizer,
                  'loss':loss,
                  'memory':memory,
                  'v_fn':v_fn,
                  'v_key':v_key,
		'device':device
                  }

    agent = DQAgent(**dqa_params)
        
    trainer_train_params = {'game':game,
                            'agent':agent,
                            'n_games':args['n_games'],
                            'n_print':args['n_print'],
                            'delay':args['delay'],
                            'curriculum':{'epoch':args['curr_epoch'], 'params':curr_params},
			'device':device}
        
    trainer = DQTrainer()
    
    ##################################################
    # F-Min
    ##################################################
    
    ##ADD IN FMIN 
       
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
                           'delay':args['delay_eval'],
			'device':device}
    
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
