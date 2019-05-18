import matplotlib.pyplot as plt

def setDefault(figsize=(20, 10)):
    plt.style.use(['dark_background', 'bmh'])
    plt.rc('axes', facecolor='k')
    plt.rc('figure', facecolor='k')
    plt.rc('figure', figsize=figsize)
    
def plotQValues(agent, value, n_states):
    keys = ["{}_{}".format(s, value) for s in range(n_states)]
    
    stops, swaps = [], []
    
    for i, key in enumerate(keys):
        
        if (agent.Q[key][0] == 0) & (agent.Q[key][1] == 0):
            continue
        else:
            stops.append((i, agent.Q[key][0]))
            swaps.append((i, agent.Q[key][1]))
    
    plt.xlim(0, n_states+1)
    plt.scatter(*zip(*stops), cmap=plt.cm.Spectral, label="Stops")
    plt.scatter(*zip(*swaps), cmap=plt.cm.Spectral, label="Swaps")
    plt.ylabel("Q-value")
    plt.xlabel("State")
    plt.title("Q-value vs. State")
    plt.legend()
    plt.show()