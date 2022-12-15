import matplotlib.pyplot as plt
import numpy as np
import neural_structure as nr



# try to recreate toxic plants example from :
# https://www.youtube.com/watch?v=hfMk-kjRv4c


def get_in_from_ou(inputs, outputs):
    """Place inputs as lists into dictionnary of all possible outputs.

    Args:
        inputs (iterable): all of the inputs (order is conserved)
        outputs (list): all of the outputs (order is conserved)

    Returns:
        dict: dictionnary containing outputs as keys and lists of lists of inputs as values.
    """
    n = len(inputs)
    states = {}
    for state in list(set(outputs)):
        states[state] = []
    for i in range(len(outputs)-1):
        states[outputs[i]].append([inputs[j][i] for j in range(len(inputs))])
    return states

def extract(iter, pos):
    """Extract lists from a list of lists.

    Args:
        iter (iterable): the list of lists
        pos (int): which coordinate to extract a list from

    Returns:
        list: list containing every element of position `pos`.
    """
    return [iter[i][pos] for i in range(len(iter))]

# ----------------------------------------------------------------------------


def generate_plants(N, rule):
    x = []
    y = []
    outputs = []
    for i in range(N+1):
        x += [np.random.randint(1,100)/10]
        y += [np.random.randint(1,100)/10]
        if rule(x[-1],y[-1]):
            outputs += [(0,1)]
        else:
            outputs += [(1,0)]
    inputs = x,y
    return inputs, outputs



network = nr.NeuralNetwork([2,3,2])

N = 100
states_names = {(1,0):"Safe", (0,1):"Toxic"}
states_colors = {(1,0):(0,0,1), (0,1):(1,0,0)}

fig, ax = plt.subplots()

rule = lambda x,y : 1 if (x+1.5*y) > 10 else 0


plants = generate_plants(N,rule)
plants = get_in_from_ou(plants[0], plants[1])


for state in plants.keys():
    for plant in plants[state]:
        dpoint = nr.DataPoint(plant, state)
        print(f"{dpoint}\n")
    ax.plot(extract(plants[state], 0), extract(plants[state], 1), "o", color=states_colors[state], label=states_names[state])

plt.legend()
plt.show()


