import numpy as np


class Game():
    def __init__(self, lo, hi, n_states, replace, reward_fn):            
        self.validateInput(lo, hi, n_states, replace, reward_fn)
        ###FIX REPLACEMENT MAX STATE
        
    def reset_game(self):
        """Reset the game states """
        self.states = np.random.choice(np.arange(self.lo, self.hi+1), size=self.n_states, replace=self.replace)
        self.max_val, self.max_state = self.states.max(), self.states.argmax()
        
    def autoTrain(self, agent, n_games):  
        """Train an agent over n_games"""
        for i in range(n_games):
            self.reset_game()
            agent.reset()
            
            state = -1
                
            while True:
                state += 1
        
                action = agent.get_action(state, self.states[state]) if state != self.n_states-1 else 0

                if (action == 0) or (state == self.n_states-1):
                    if state == self.max_state:
                        reward = self.reward_fn(True)
                        agent.wins += 1
                    else:
                        reward = self.reward_fn(False)
                    break
                else:
                    reward = 0

                action_ = agent.get_action(state+1, self.states[state+1])

                agent.update(state, action, state+1, action_, reward)

            agent.rewards.append(reward)
            agent.final_state.append(state)
            agent.update(state, action, None, None, reward)
                
        
    def validateInput(self, lo, hi, n_states, replace, reward_fn):
        assert(type(lo) == int), "lo must be an int"
        assert(type(hi) == int), "hi must be an int"
        assert(type(n_states) == int), "n_states must be an int"
        assert(type(replace) == bool), "replace must be a bool"
        assert(callable(reward_fn)), "reward_fn must be a function"
        assert(lo < hi), "lo must be strictly smaller than hi"
        assert((hi + 1 - lo) >= n_states), "n_cards must be less than or equal to the range from lo to hi"
        
        self.lo, self.hi, self.n_states, self.replace, self.reward_fn  = lo, hi, n_states, replace, reward_fn