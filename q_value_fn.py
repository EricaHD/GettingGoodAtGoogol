import numpy as np

def QkeyMaxV(s, v, q_key):
    if v is not None:
        prev_v = int(q_key.split("_")[1])

        if v > prev_v:
            return "{}_{}".format(s, v)
        else:
            return "{}_{}".format(s, prev_v)
    else:
        return "{}_{}".format(s, int(q_key.split("_")[1]))