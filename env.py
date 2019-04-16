import os 
from time import sleep

import numpy as np
from tqdm import tqdm

from IPython.display import clear_output

from game import Game

class Env():
    def __init__(self):
        return
        
    def train(self, game, agent, n_games, reward_fn, n_print, delay):
        """Train an agent over n_games"""
        
        wins = 0
        
        #Iterate through games
        for i in tqdm(range(n_games)):
            #Reset game state and agent state
            game.reset()
            agent.reset()
            
            self.params = self.reset(game)
            self.params['i'] = i
            
            #Iterate through game
            while True:
                
                self.params['action'] = agent.getAction(self.params) 
                      
                #Check for stop
                if (self.params['action'] == 0) or (game.winState()):
                    #Check for win
                    self.params['reward'], win = reward_fn(game, self.params)
                    break
                else:
                    self.params['reward'] = 0
                
                #Update Q-values
                
                self.params['val'] = game.flip()
                self.params['state'] = game.state
                
                
                agent.update(self.params)
                
                
            #Update Q-values 
            self.params['game_over'] = True
            agent.update(self.params)
            
            if (i%n_print == 0) & (i > 0):
                sleep(delay)
                clear_output()
                print("GAME: {} | VICTORY PERCENTAGE: {:.2}".format(i, wins/i))
                    
            wins += win
        clear_output()
        print("TRAINING COMPLETE | FINAL VICTORY PERCENTAGE: {:.2}".format(wins/n_games))
            
    def eval(self, game, agent, n_games, reward_fn, n_print, delay):
        """Train an agent over n_games"""
        
        wins = 0
        
        #Iterate through games
        for i in tqdm(range(n_games)):
            #Reset game state and agent state
            game.reset()
            agent.reset()
            
            self.params = self.reset(game)
            
            #Iterate through game
            while True:
                
                self.params['action'] = agent.getAction(self.params) 
                      
                #Check for stop
                if (self.params['action'] == 0) or (game.winState()):
                    #Check for win
                    self.params['reward'], win = reward_fn(game, self.params)
                    break
                else:
                    self.params['reward'] = 0
                
                #Update Q-values
                
                self.params['val'] = game.flip()
                self.params['state'] = game.state
                
                
            #Update Q-values 
            self.params['game_over'] = True
            
            if (i%n_print == 0) & (i > 0):
                sleep(delay)
                clear_output()
                print("GAME: {} | VICTORY PERCENTAGE: {:.2}".format(i, wins/i))
                    
            wins += win
        
        clear_output()
        print("EVAL COMPLETE | FINAL VICTORY PERCENTAGE: {:.2}".format(wins/n_games))
    
    def reset(self, game): 
        
        return {"lo":game.lo,
                "hi":game.hi,
                "n_states":game.n_states,
                "replace":game.replace,
                "state":0,
                "action":None,
                "val":game.val,
                "game_over":False}
    