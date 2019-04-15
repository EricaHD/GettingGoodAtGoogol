import numpy as np
from tqdm import tqdm

class Game():
    def __init__(self, lo, hi, n_states, replace, reward_fn):            
        self.validateInput(lo, hi, n_states, replace, reward_fn)
        
    def train(self, agent, n_games=10_000, save_results=False, teacher=False):  
        """Train an agent over n_games"""

        agent_wins = 0
        final_states, final_values = [], []
        
        #Iterate through games
        for i in tqdm(range(n_games)):
            #Reset game state and agent state
            self.resetGame()
            agent.resetGame()
            
            state = -1
                
            #Iterate through game
            while True:
                state += 1
                self.params['state'] = state
                self.params['val'] = self.states[state]
                
                #Get action, if at end, only allow one action
                if not teacher:
                    action = agent.getAction(self.params) 
                else:
                    action = teacher.getAction(self.params) 
                    agent.observe(self.params)
                      
                self.params['action'] = action

                #Check for stop
                if (action == 0) or (state == self.params['n_states']-1):
                    #Check for win
                    val_rank = np.where(self.params['val'] == self.states_sorted)[0][0]
                    self.params['reward'], win = self.reward_fn(val_rank)
                    break
                else:
                    self.params['reward'] = 0
                
                #Get next action
                if not teacher:
                    action_ = agent.getAction(self.params)
                else:
                    action_ = teacher.getAction(self.params) 
                    agent.observe(self.params)

                #Update Q-values
                self.params['action_'] = self.params['action']
                agent.update(self.params)
            
            #Update Q-values 
            self.params['state_'], self.params['action_'] = None, None
            agent.update(self.params)
            
            if save_results:
                agent_wins += win
                final_states.append((state, self.params['max_state']))
                final_values.append((self.states[state], self.params['max_val']))
            
        if save_results:
            return agent_wins, final_states, final_values        
            
    def eval(self, agent, n_games):  
        """Let agent play without training"""
        
        agent_wins = 0
        final_states, final_values = [], []
        
        #Iterate through games
        for i in tqdm(range(n_games)):
            #Reset game state and agent state
            self.resetGame()
            agent.resetGame()
            
            state = -1
                
            #Iterate through game
            while True:
                state += 1
                self.params['state'] = state
                self.params['val'] = self.states[state]
                
                #Get action, if at end, only allow one action
                action = agent.getAction(self.params) 

                #Check for stop
                if (action == 0) or (state == self.params['n_states']-1):
                    #Check for win
                    val_rank = np.where(self.params['val'] == self.states_sorted)[0][0]
                    reward, win = self.reward_fn(val_rank)
                    break
                else:
                    self.params['reward'] = 0
            
            agent_wins += win
            final_states.append((state, self.params['max_state']))
            final_values.append((self.states[state], self.params['max_val']))
            
        return agent_wins, final_states, final_values        
        
    def validateInput(self, lo, hi, n_states, replace, reward_fn):
        assert(type(lo) == int), "lo must be an int"
        assert(type(hi) == int), "hi must be an int"
        assert(type(n_states) == int), "n_states must be an int"
        assert(type(replace) == bool), "replace must be a bool"
        assert(callable(reward_fn)), "reward_fn must be a function"
        assert(lo < hi), "lo must be strictly smaller than hi"
        assert((hi + 1 - lo) >= n_states), "n_cards must be less than or equal to the range from lo to hi"
        
        self.reward_fn  = reward_fn
                      
        self.params = {'lo':lo,
                       'hi':hi,
                       'n_states':n_states,
                       'replace':replace,
                       'state':None,
                       'action':None,
                       'state_':None,
                       'action_':None}              
        
    def resetGame(self):
        """Reset the game states """
        self.states = np.random.choice(np.arange(self.params['lo'], self.params['hi']+1), 
                                       size=self.params['n_states'], 
                                       replace=self.params['replace'])
        self.states_sorted = np.sort(self.states) 
        
        self.params['max_val'] = self.states.max()
        self.params['max_state'] = self.states.argmax()
