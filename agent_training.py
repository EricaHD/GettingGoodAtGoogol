import random
from argparse import ArgumentParser

import numpy as np

from game import Game
from agent import QAgent, OptimalAgent
from utils import qKeyMaxBin, simpleReward

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
    ap.add_argument("-r", "--reward", default="simpleReward",
                    help="Reward Fn")
    
    #Training Parameters
    ap.add_argument("-ng", "--n_games", default=10_000,
                    help="N-Games")
    
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
    ap.add_argument("-qkf", "--q_key_fn", default="bin12",
                    help="Q-Key Fn")
    ap.add_argument("-qk", "--q_key", default="0_0",
                    help="Q-Key")
    
    #Save Path
    ap.add_argument("-fp", "--file_path",
                    help="Save File Path")
    
    
    args = vars(ap.parse_args())
    
    if args['reward'] == "simpleReward":
        reward_fn = simpleReward
    
    game = Game(lo=int(args['lo']), hi=int(args['hi']), n_states=int(args['n_states']), replace=bool(args['replace']), reward_fn=reward_fn)
    
    if args['agent'] == "q_learn":
        if args['q_key_fn'] == "bin22":
            q_key_fn = lambda s, u, p, q: qKeyMaxBin(s, u, p, q, 2, 2)
        elif args['q_key_fn'] == "bin12":
            q_key_fn = lambda s, u, p, q: qKeyMaxBin(s, u, p, q, 1, 2)
        elif args['q_key_fn'] == "bin21":
            q_key_fn = lambda s, u, p, q: qKeyMaxBin(s, u, p, q, 2, 1)
        elif args['q_key_fn'] == "bin11":
            q_key_fn = lambda s, u, p, q: qKeyMaxBin(s, u, p, q, 1, 1)
 
        agent_params = {'alpha':float(args['alpha']),
                      	'gamma':float(args['gamma']),
                      	'eps':float(args['epsilon']), 
                      	'eps_decay':float(args['eps_decay']), 
                      	's_cost':float(args['s_cost']),
                      	'sarsa':bool(args['q_learn']),
                      	'q_key_fn':q_key_fn,
                      	'q_key':args['q_key']}
        
        agent = QAgent(**agent_params)
    
    print("Beginning training")            
    agent_wins, _, _ = game.autoTrain(agent, int(args['n_games']), True, False)
    
    print("Training complete, winning percentage: {:.2}".format(agent_wins/int(args['n_games'])))
    
    with open(args['file_path'], 'wb') as file:
        pkl.dump(dict(agent.Q), file)
        
    print("Agent Saved at: {}".format(args['file_path']))
