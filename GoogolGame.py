import numpy as np


class Game():
    def __init__(self, lo, hi, n_states, replace, reward):            
        self.validateInput(lo, hi, n_states, replace, reward)
        
        self.states = np.random.choice(np.arange(lo, hi+1), size=n_states, replace=replace)
        self.max_val, self.max_state = self.states.max(), self.states.argmax()
        
        ###FIX REPLAECEMENT MAX STATE
        
    def autoTrain(self, agent):
        
        state = -1
        
        while True:
            state += 1
            action = agent.get_action(state)
            
            if (action == 0) or (state == self.n_states):
                reward = self.reward * (state == self.max_state) 
                break
            else:
                reward = 0
        
            action_ = agent.get_action(state+1)
            
            agent.update(state, action, state+1, action_, reward)
            #agent.rewards.append(reward)
            
        agent.rewards.append(reward)
        agent.final_state.append(state)
        agent.update(state, action, None, None, reward)
                
        
    def validateInput(self, lo, hi, n_states, replace, reward):
        assert(type(lo) == int), "lo must be an int"
        assert(type(hi) == int), "hi must be an int"
        assert(type(n_states) == int), "n_states must be an int"
        assert(type(replace) == bool), "replace must be a bool"
        assert(type(reward) == int), "verbose must be a int"
        assert(lo < hi), "lo must be strictly smaller than hi"
        assert((hi + 1 - lo) >= n_states), "n_cards must be less than or equal to the range from lo to hi"
        
        self.lo, self.hi, self.n_states, self.replace, self.reward  = lo, hi, n_states, replace, reward