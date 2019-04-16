import os 

import numpy as np
from tqdm import tqdm

from game import Game

class Env():
    def __init__(self):
        return
        
    def train(self, game, agent, n_games, reward_fn, verbose):
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
                
                
                agent.update(self.params)
                
                
            #Update Q-values 
            self.params['game_over'] = True
            agent.update(self.params)
            
            #if verbose:
            #    if int(i/n_games * 100) :
            #        print("Current Training Score: {:.2}".format(wins/i))
                    
            wins += win
        
        if verbose:
            print("Training complete; Agent won {:.2}% of games".format(wins/n_games))
    
    def reset(self, game): 
        
        return {"lo":game.lo,
                "hi":game.hi,
                "n_states":game.n_states,
                "replace":game.replace,
                "state":0,
                "action":None,
                "val":game.val,
                "game_over":False}
    