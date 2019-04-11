import numpy as np


class Game():
    def __init__(self, lo, hi, n_states, replace, reward_fn):            
        self.validateInput(lo, hi, n_states, replace, reward_fn)
        
    def autoTrain(self, agent, n_games, save_results):  
        """Train an agent over n_games"""
        
        agent_wins = 0
        final_states, final_values = [], []
        
        #Iterate through games
        for i in range(n_games):
            #Reset game state and agent state
            self.resetGame()
            agent.resetGame()
            
            state = -1
                
            #Iterate through game
            while True:
                state += 1
        
                #Get action, if at end, only allow one action
                action = agent.getAction(state, self.states[state]) if state != self.n_states-1 else 0

                #Check for stop
                if (action == 0) or (state == self.n_states-1):
                    #Check for win
                    if self.states[state] == self.max_val:
                        reward = self.reward_fn(True)
                        agent.wins += 1
                    else:
                        reward = self.reward_fn(False)
                    break
                else:
                    reward = 0
                
                #Get next action
                action_ = agent.getAction(state+1, self.states[state])

                #Update Q-values
                agent.update(state, action, state+1, action_, reward)
            
            #Update Q-values 
            agent.update(state, action, None, None, reward)
            
            if save_results:
                agent_wins += win
                final_states.append((state, self.max_state))
                final_values.append((self.states[state], self.max_val))
            
        if save_results:
            return agent_wins, final_states, final_values
            
    def autoTeach(self, agent, teacher, n_games):  
        """Teach a learning agent over n_games"""
        
        #Iterate through games
        for i in range(n_games):
            #Reset game state, agent state and teacher state
            self.resetGame()
            agent.resetGame()
            teacher.resetGame()
            
            state = -1
                
            #Iterate through games
            while True:
                state += 1
        
                #Get teacher actions
                action = teacher.getAction(state, self.states[state])
                #Add key to agent 
                agent.observe(state, self.states[state])

                #Check for stop
                if (action == 0) or (state == self.n_states-1):
                    #Check for victory
                    if state == self.max_state:
                        reward = self.reward_fn(True)
                    else:
                        reward = self.reward_fn(False)
                    break
                else:
                    reward = 0

                #Get teachers next action
                action_ = teacher.getAction(state+1, self.states[state])
                #Add key to agent 
                agent.observe(state+1, self.states[state])
        
                #Update agent
                agent.update(state, action, state+1, action_, reward)
            
            #Update agent
            agent.update(state, action, None, None, reward)
        
            
    def autoPlay(self, agent, n_games):  
        """Let agent play without training"""
        
        agent_wins = 0
        final_states, final_values = [], []
        
        #Iterate through games
        for i in range(n_games):
            #Reset game state and agent state
            self.resetGame()
            agent.resetGame()
            
            state = -1
                
            #Iterate through game
            while True:
                state += 1
        
                #Get action, if at end, only allow one action
                action = agent.getAction(state, self.states[state]) if state != self.n_states-1 else 0

                #Check for stop
                if (action == 0) or (state == self.n_states-1):
                    #Check for win
                    if self.states[state] == self.max_val:
                        reward = self.reward_fn(True)
                        win = 1
                    else:
                        reward = self.reward_fn(False)
                        win = 0
                    break
                else:
                    reward = 0
            
            agent_wins += win
            final_states.append((state, self.max_state))
            final_values.append((self.states[state], self.max_val))
            
        return agent_wins, final_states, final_values
        
    def validateInput(self, lo, hi, n_states, replace, reward_fn):
        assert(type(lo) == int), "lo must be an int"
        assert(type(hi) == int), "hi must be an int"
        assert(type(n_states) == int), "n_states must be an int"
        assert(type(replace) == bool), "replace must be a bool"
        assert(callable(reward_fn)), "reward_fn must be a function"
        assert(lo < hi), "lo must be strictly smaller than hi"
        assert((hi + 1 - lo) >= n_states), "n_cards must be less than or equal to the range from lo to hi"
        
        self.lo, self.hi, self.n_states, self.replace, self.reward_fn  = lo, hi, n_states, replace, reward_fn
        
    def resetGame(self):
        """Reset the game states """
        self.states = np.random.choice(np.arange(self.lo, self.hi+1), size=self.n_states, replace=self.replace)
        self.max_val, self.max_state = self.states.max(), self.states.argmax()
        