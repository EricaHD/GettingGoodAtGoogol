import numpy as np

import gzip
import cloudpickle as pkl

#########################################################################################
##Q-KEY FN
#########################################################################################

def qKeyMaxBin(params, idx, v, i_d, v_d):
    i_key = int(np.round(idx/params['n_idx'], i_d)*100)
    v_key = int(np.round(v/params['hi'], v_d)*100)
    
    return str((i_key, v_key))

def vMax(params, v, v_):
    if v > v_:
        return v
    else:
        return v_
    
def vSeq(params, v, v_):
    seq = eval(v_)
    seq.append(v)
    return str(seq)

def qKeySeq(params, idx, v, v_d):
    seq = eval(v)
    seq[idx] = int(np.round(seq[idx]/params['hi'], v_d)*100)
    return str(seq)

#########################################################################################
##REWARD FN
#########################################################################################
    
    
def rewardScalar(game, pos_reward, neg_reward):
    if game.val == game.max_val:
        return pos_reward, 1
    else:
        return neg_reward, 0 

def rewardTopN(game, pos_reward, neg_reward, n_pct):
    rank = np.where(game.val == game.values_sorted)[0][0]
    
    if rank < n_pct:
        return pos_reward - rank, 1
    else:
        return np.maximum(neg_reward, -rank) , 0

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
