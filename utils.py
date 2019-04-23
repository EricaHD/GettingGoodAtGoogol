from itertools import product

import numpy as np

import gzip
import cloudpickle as pkl

#########################################################################################
##Q-KEY FN
#########################################################################################

def qKeyMaxBin(params, idx, v, agent_params):
    i_key = int(np.round(idx/params['n_idx'], agent_params['i_bin'])*100)
    v_key = int(np.round(v/params['hi'], agent_params['v_bin'])*100)
    
    return str((i_key, v_key))

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

#########################################################################################
##REWARD FN
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
        return np.maximum(game.reward['neg'], -rank) , 0

#########################################################################################
##SAVE AND LOAD FN
#########################################################################################

def svZipPkl(obj, filename):
	with gzip.open(filename, 'wb') as f:
		pkl.dump(obj, f)

def ldZipPkl(filename):
	with gzip.open(filename, 'rb') as f:
		obj = pkl.load(f)
		return obj
    
#########################################################################################
##OP FN
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
