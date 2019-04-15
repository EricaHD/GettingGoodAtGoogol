import numpy as np

def qKeyMaxBin(s, update, params, q_key, s_d, v_d):
    
    s_key = int(np.round(s/params['n_states'], s_d)*100)
    
    if not update:
        v = params['val']
        
        prev_v = int(q_key.split("_")[1])
        
        if v > prev_v:
            
            v_key = int(np.round(v/params['max_val'], v_d)*100)
            
            return "{}_{}".format(s_key, v_key)
        else:
            return "{}_{}".format(s_key, prev_v)
    else:
        return "{}_{}".format(s_key, int(q_key.split("_")[1]))
    
def simpleReward(val_rank):
    if val_rank == 0:
	return 10, 1
    else:
	return -1, 0 

def topTenReward(val_rank):
    if val_rank < 10:
	return 10 - val_rank
    else:
        return -val_rank
