import numpy as np

class Game():
    def __init__(self, lo, hi, n_cards, replace, verbose):            
        self.validateInput(lo, hi, n_cards, replace, verbose)
        
        self.cards = np.random.choice(np.arange(lo, hi+1), size=n_cards, replace=replace)
        self.max_card, self.max_pos = self.cards.max(), self.cards.argmax()
        
        self.pos = -1
        
        self.verbose = verbose
        self.game_over = False
        
    def step(self, action):
        assert(type(action) == bool), "action must be a boolean"
        if self.game_over: return 
        
        if action:
            self.pos += 1
            return self.cards[self.pos]
        else:
            reward = self.pos == self.max_pos
            if self.verbose & reward:
                print("Winner!")
            elif self.verbose & ~reward:
                print("Winning Card position: {} | value: {}".format(self.max_pos, self.max_card))
            
            self.game_over = True
            
            return reward, self.max_pos, self.max_card
        
    def reset(self):
        self.cards = np.random.choice(np.arange(self.lo, self.hi+1), size=self.n_cards, replace=self.replace)
        self.max_card, self.max_pos = self.cards.max(), self.cards.argmax()
        
        self.pos = -1
        self.game_over = False
        
    def validateInput(self, lo, hi, n_cards, replace, verbose):
        assert(type(lo) == int), "lo must be an int"
        assert(type(hi) == int), "hi must be an int"
        assert(type(n_cards) == int), "n_cards must be an int"
        assert(type(replace) == bool), "replace must be a bool"
        assert(type(verbose) == bool), "verbose must be a bool"
        assert(lo < hi), "lo must be strictly smaller than hi"
        assert((hi + 1 - lo) >= n_cards), "n_cards must be less than or equal to the range from lo to hi"
        
        self.lo, self.hi, self.n_cards, self.replace, self.verbose = lo, hi, n_cards, replace, verbose