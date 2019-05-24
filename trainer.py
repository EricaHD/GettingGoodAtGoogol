import os 
from time import time, sleep

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from game import Game
from util import *

from IPython.display import clear_output

#########################################################################################
# Basic Trainer
#########################################################################################

class Trainer:
    def __init__(self):
        return
    
    def train(self, game, agent):
        pass
    
    def eval(self, game, agent, n_games, n_print, delay):
        """Train an agent over n_games"""
        
        wins = 0
        stop_choices = [0] * game.n_idx
        agent.eval()
        
        # Iterate through games
        for i in tqdm(range(n_games), leave=False):
            # Reset game and agent
            game.reset()
            agent.reset()
            
            self.params = self.reset(game)
            
            # Iterate through game
            while True:
                
                action = agent.getAction(self.params) 
                if action == 0:
                    stop_choices[self.params['idx']] += 1
                self.params['idx'], self.params['val'], self.params['reward'], self.params['game_status'] = game.step(action)
                
                if self.params['game_status']:
                    wins += game.win
                    break
            
            if (i % n_print == 0) & (i > 0):
                sleep(delay)
                clear_output()
                print("EVAL PCT: {:.2} |\t VICTORY PERCENTAGE: {:.2}".format(i/n_games, wins/i))
        
        clear_output()
        print("EVAL COMPLETE |\t FINAL VICTORY PERCENTAGE: {:.2}".format(wins/n_games))
        return wins/n_games, stop_choices
        
    def reset(self, game): 
        
        return {"lo":game.lo,
                "hi":game.hi,
                "n_idx":game.n_idx,
                "replace":game.replace,
                "idx":0,
                "action":None,
                "val":game.val,
                "game_status":False}

#########################################################################################
# Q-Learner Trainer
#########################################################################################

class QTrainer(Trainer):
    def __init__(self):
        super().__init__()
        return

    def train(self, game, agent, n_games, n_print, delay):
        """Train an agent over n_games"""
        
        wins, games = 0, 0
        agent.train()
        
        # Iterate through games
        for i in tqdm(range(n_games), leave=False):
            #Reset game and agent
            game.reset()
            agent.reset()
            
            self.params = self.reset(game)
            self.params['game_i'] = i
            
            # Iterate through game
            while True:
                
                self.params['action'] = agent.getAction(self.params) 
                self.params['idx'], self.params['val'], self.params['reward'], self.params['game_status'] = game.step(self.params['action'])
                
                if self.params['game_status']:
                    wins += game.win
                    break
                else:
                    #Step and update
                    self.params['action'] = agent.getAction(self.params) 
                    self.params['idx'], self.params['val'], _, _ = game.step(self.params['action'])
                    agent.update(self.params)
                
            # Update Q-values 
            agent.update(self.params)
            
            games += 1
            
            if (i % n_print == 0) & (i > 0):
                sleep(delay)
                clear_output()
                print("TRAIN PCT: {:.2} |\t VICTORY PERCENTAGE: {:.2}".format(i/n_games, wins/games))
                 
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
                "game_status":False}
    
#########################################################################################
# MCMC Trainer
#########################################################################################

class MCMCTrainer(Trainer):
    def __init__(self):
        super().__init__()
        return
        
    def train(self, game, agent, n_games):
        
        wins, games = 0, 0
        agent.train()
        
        for i in tqdm(range(n_games), leave=False):
            game.reset()
            agent.reset()
            self.params = self.reset(game)
            
            episode = self.mcEpisode(game, agent)
            
            if episode[-1][3] > 0:  # if the reward was positive, we probably won
                wins += 1
            games += 1
            
            agent.update(self.params, episode)
        
        return wins/games
    
    def mcEpisode(self, game, agent):
        action, val = agent.getAction(self.params)
        self.params['idx'], self.params['val'], reward, self.params['game_status'] = game.step(action)
        
        if self.params['game_status']:
            return [[self.params['idx'], val, action, reward]] 
        else:
            return [[self.params['idx']-1, val, action, reward]] + self.mcEpisode(game, agent)
        
    def reset(self, game): 
        
        return {"lo":game.lo,
                "hi":game.hi,
                "n_idx":game.n_idx,
                "replace":game.replace,
                "idx":game.idx,
                "val":game.val}
    
#########################################################################################
# DQN Trainer
#########################################################################################

class DQTrainer(Trainer):
    def __init__(self):
        super().__init__()
        return
    
    def train(self, game, agent, n_games, n_print, delay, device):
        """Train a DQAgent over n_games"""
        
        wins, games = 0, 0
        agent.train()
        
        for game_i in tqdm(range(n_games)):
            game.reset()
            agent.reset()

            self.params = self.reset(game)
            self.params['game_i'] = game_i

            while True:
                action, state = agent.getAction(self.params)
                self.params['idx'], self.params['val'], self.params['reward'], self.params['game_status'] = game.step(action.item())
                reward = torch.tensor([self.params['reward']], device=device)

                if self.params['game_status']:
                    next_state = None
                    agent.memory.push(state, action, next_state, reward)
                    agent.update()
                    wins += game.win
                    break
                else:
                    next_state = agent.p_to_s(self.params, agent.v_key)
                    agent.memory.push(state, action, next_state, reward)
                    state = next_state
                    agent.update()

            agent.updateNet(game_i)
            games += 1

            if (game_i % n_print == 0) & (game_i > 0):
                sleep(delay)
                clear_output()
                print("TRAIN PCT: {:.2} |\t VICTORY PERCENTAGE: {:.2}".format(game_i/n_games, wins/games))
                
        clear_output()
        print("TRAINING COMPLETE |\t FINAL VICTORY PERCENTAGE: {:.2}".format(wins/games))
        return wins/games
    
    def eval(self, game, agent, n_games, n_print, delay, device):
        """Eval a DQAgent over n_games"""
        
        wins, games = 0, 0
        stop_choices = [0] * game.n_idx
        agent.eval()
        
        for game_i in tqdm(range(n_games)):
            game.reset()
            agent.reset()

            self.params = self.reset(game)
            self.params['game_i'] = game_i

            while True:
                action = agent.getAction(self.params)
                if action == 0:
                    stop_choices[self.params['idx']] += 1
                self.params['idx'], self.params['val'], self.params['reward'], self.params['game_status'] = game.step(action.item())
                reward = torch.tensor([self.params['reward']], device=device)

                if self.params['game_status']:
                    wins += game.win
                    break
                    
            games += 1

            if (game_i % n_print == 0) & (game_i > 0):
                sleep(delay)
                clear_output()
                print("TRAIN PCT: {:.2} |\t VICTORY PERCENTAGE: {:.2}".format(game_i/n_games, wins/games))
                       
        clear_output()
        print("TRAINING COMPLETE |\t FINAL VICTORY PERCENTAGE: {:.2}".format(wins/games))
        print("STOP CHOICES:", stop_choices)
        return wins/games, stop_choices
    
    def reset(self, game): 
        
        return {"lo":game.lo,
                "hi":game.hi,
                "n_idx":game.n_idx,
                "replace":game.replace,
                "idx":0,
                "action":None,
                "val":game.val,
                "game_status":False}
