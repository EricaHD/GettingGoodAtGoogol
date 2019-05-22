import random 

from collections import namedtuple
from itertools import product

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import gzip
import cloudpickle as pkl


#########################################################################################
# Q-KEY FN
#########################################################################################

def qKeyMaxBin(params, idx, v, agent_params):
    i_key = int(np.round(idx/params['n_idx'], agent_params['i_bin'])*100)
    v_key = int(np.round(v/params['hi'], agent_params['v_bin'])*100)
    
    return str((i_key, v_key))


def qKeyMaxBinV(params, idx, v, agent_params):
    i_key = int(np.round(idx/params['n_idx'], agent_params['i_bin'])*100)
    val_key = int(np.round(params['val']/params['hi'], agent_params['v_bin'])*100)
    v_key = int(np.round(v/params['hi'], agent_params['v_bin'])*100)
    
    return str((i_key, val_key, v_key))


def vMax(params, v, v_):
    if v > v_:
        return v
    else:
        return v_


def qKeySeq(params, idx, v, agent_params):
    seq = eval(v)
    seq[idx] = int(np.round(seq[idx]/params['hi'], agent_params['v_bin'])*100)
    return str(seq)


def vSeq(params, v, v_):
    seq = eval(v_)
    seq.append(v)
    return str(seq)


def vIdx(params, v, v_):
    return v


def stateMax(params, v_key):
    return torch.tensor([params['idx']/params['n_idx'], v_key/params['hi']]).unsqueeze(0)


def stateMaxV(params, v_key):
    return torch.tensor([params['idx']/params['n_idx'], params['val']/params['hi'], v_key/params['hi']]).unsqueeze(0)


#########################################################################################
# REWARD FN
#########################################################################################
    
def rewardScalar(game):
    if game.val == game.max_val:
        return game.reward['pos'], 1
    else:
        return game.reward['neg'], 0 


def rewardTopN(game):
    rank = np.where(game.val == game.values_sorted)[0][0]
    
    n_pct = int(game.reward['n']/100 * game.n_idx)
    
    if rank <= n_pct:
        return game.reward['pos'] - rank, 1
    else:
        return np.maximum(game.reward['neg'], -rank), 0


#########################################################################################
# SAVE AND LOAD FN
#########################################################################################

def svZipPkl(obj, filename):
    with gzip.open(filename, 'wb') as f:
        pkl.dump(obj, f)


def ldZipPkl(filename):
    with gzip.open(filename, 'rb') as f:
        obj = pkl.load(f)
        return obj


#########################################################################################
# OP FN
#########################################################################################

def convertOp(op):
    if op == "minus":
        return '-'
    elif op == "plus":
        return "+"
    elif op == "times":
        return "*"
    elif op == "divide":
        return "/"
    
    return op


#########################################################################################
# DQN Memory
#########################################################################################

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


Transition = namedtuple('Transition',
                       ('state', 'action', 'next_state', 'reward'))
