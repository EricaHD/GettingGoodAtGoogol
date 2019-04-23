import os 
from time import time, sleep

import numpy as np
from tqdm import tqdm

from IPython.display import clear_output

from game import Game

#########################################################################################
##Basic Trainer
#########################################################################################

class Trainer():
    def __init__(self):
        return
    
    def train(self, game, agent):
        pass
    
    def eval(self, game, agent, n_games, n_print, delay):
        """Train an agent over n_games"""
        
        wins = 0
        agent.eval()
        
        
        
        #Iterate through games
        for i in tqdm(range(n_games), leave=False):
            #Reset game and agent
            game.reset()
            agent.reset()
            
            self.params = self.reset(game)
            
            #Iterate through game
            while True:
                
                self.params['action'] = agent.getAction(self.params) 
                      
                #Check for stop
                if self.params['action'] == 0:
                    self.params['reward'], win = game.getReward()
                    game.setGameStatus()
                    break
                else:
                    self.params['reward'] = 0
                
                self.params['idx'], self.params['val'] = game.step()
            
            wins += win
            
            if (i%n_print == 0) & (i > 0):
                sleep(delay)
                clear_output()
                print("EVAL PCT: {:.2} |\t VICTORY PERCENTAGE: {:.2}".format(i/n_games, wins/i))
            
        
        clear_output()
        print("EVAL COMPLETE |\t FINAL VICTORY PERCENTAGE: {:.2}".format(wins/n_games))
        
    def reset(self):
        pass

#########################################################################################
##Q-Learner Trainer
#########################################################################################

class QTrainer(Trainer):
    def __init__(self):
        super().__init__()
        return

    def train(self, game, agent, n_games, n_print, delay, curriculum):
        """Train an agent over n_games"""
        
        wins, games = 0, 0
        agent.train()
        
        #Iterate through games
        for i in tqdm(range(n_games), leave=False):
            #Reset game and agent
            game.reset()
            agent.reset()
            
            self.params = self.reset(game)
            self.params['game_i'] = i
            
            #Iterate through game
            while True:
                
                self.params['action'] = agent.getAction(self.params) 
                      
                #Check for stop
                if self.params['action'] == 0:
                    self.params['reward'], win = game.getReward()
                    break
                else:
                    self.params['reward'] = 0
                
                #Step and update
                self.params['idx'], self.params['val'] = game.step()
                agent.update(self.params)
                
            #Update Q-values 
            self.params['game_over'] = True
            agent.update(self.params)
            
            wins += win
            games += 1
            
            if (i%n_print == 0) & (i > 0):
                sleep(delay)
                clear_output()
                print("TRAIN PCT: {:.2} |\t VICTORY PERCENTAGE: {:.2}".format(i/n_games, wins/games))
                
                
            if (i%curriculum['epoch'] == 0) & (i > 0):
                wins, games = 0, 0
                for k, v in game.reward.items():
                    v_ = eval("{} {} {}".format(v, curriculum['params']['op'], curriculum['params'][k]))
                    game.reward[k] = v_
                    
                print("ADJUSTING REWARDS")
                
                
                    
        clear_output()
        print("TRAINING COMPLETE |\t FINAL VICTORY PERCENTAGE: {:.2}".format(wins/games))
    
        return wins/games
    
    def reset(self, game): 
        
        return {"lo":game.lo,
                "hi":game.hi,
                "n_idx":game.n_idx,
                "replace":game.replace,
                "idx":0,
                "action":None,
                "val":game.val,
                "game_over":False}
    
#########################################################################################
##MCMC Trainer
#########################################################################################

class MCMCTrainer(Trainer):
    def __init__(self):
        super().__init__()
        return
        
    def train(self, game, agent, n_episodes, curriculum):
        
        agent.train()
        
        for i in tqdm(range(n_episodes), leave=False):
            game.reset()
            agent.reset()
            self.params = self.reset(game)
            
            episode = self.mcEpisode(game, agent)
            
            agent.update(self.params, episode)
            
            if (i%curriculum['epoch'] == 0) & (i > 0):
                for k, v in game.reward.items():
                    v_ = eval("{} {} {}".format(v, curriculum['params']['op'], curriculum['params'][k]))
                    game.reward[k] = v_
                    print(k, v, v_)
                    
                print("ADJUSTING REWARDS")
    
    def mcEpisode(self, game, agent):
        action, val = agent.getAction(self.params)
        if action == 0:
            reward = game.getReward()[0]
            return [[self.params['idx'], val, action, reward]] 
        else:
            reward = 0
            self.params['idx'], self.params['val'] = game.step()
            return [[self.params['idx']-1, val, action, reward]] + self.mcEpisode(game, agent)
        
    def reset(self, game): 
        
        return {"lo":game.lo,
                "hi":game.hi,
                "n_idx":game.n_idx,
                "replace":game.replace,
                "idx":game.idx,
                "val":game.val}
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
     
        
 
