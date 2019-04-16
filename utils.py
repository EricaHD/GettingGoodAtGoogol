import numpy as np

import gzip
import cloudpickle as pkl

#########################################################################################
##Q-Key FN
#########################################################################################

def qKeyMaxBin(params, q_key, s_d, v_d):
    s_key = int(np.round(params['state']/params['n_states'], s_d)*100)
    
    v = params['val']    
    prev_v = int(q_key.split("_")[1])
        
    if v > prev_v:
            
        v_key = int(np.round(v/params['hi'], v_d)*100)
            
        return "{}_{}".format(s_key, v_key)
    else:
        return "{}_{}".format(s_key, prev_v)
    
    
def qKeySeqBin(params, q_key, v_d):
    v = params['val']
    v_key = int(np.round(v/params['hi'], v_d)*100)
    return q_key + "_{}".format(v_key)     
    
#########################################################################################
##REWARD FN
#########################################################################################
    
    
def rewardScalar(game, game_params, pos_reward, neg_reward):
    if game_params['val'] == game.max_val:
        return pos_reward, 1
    else:
        return neg_reward, 0 

def rewardTopN(game, game_params, pos_reward, neg_reward, n):
    rank = np.where(game_params['val'] == game.states_sorted)[0][0]
    
    if rank < n:
        return n - rank, 1
    else:
        return np.maximum(neg_reward, -rank) , 0

###########################################################################################
###########################################################################################

def saveZippedPkl(obj, filename):
	with gzip.open(filename, 'wb') as f:
		pkl.dump(obj, f)

def loadZippedPkl(filename):
	with gzip.open(filename, 'rb') as f:
		obj = pkl.load(f)
		return obj
