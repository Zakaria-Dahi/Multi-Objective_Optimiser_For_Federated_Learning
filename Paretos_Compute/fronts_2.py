
import json
import numpy as np
import math
import matplotlib.pyplot as plt

# data of the fully-connected newtrok
# data1[execution][iteration][rows][cols]) 

with open('fronts_dense.json') as f:
    data=json.load(f)


# import the data
data1 = data['paretos']
data2 = data['opt']


# recover the number of executions and iterations
exe= len(data1)
iter = len(data1[0])

# declare an array to stock the values of the hypervolume
x = np.linspace(1,iter,iter)

# convert data2 into an np array
data2 = np.array(data2)

# coordinates of the nidle
epsilon = 0.1 
nadir_y = np.max(data2[:,1]) + epsilon
nadir_x = np.min(data2[:,0]) + epsilon

# go through the fronts of iter
for e in range(3):
    # declare the axis of plot y
    y = np.array([0])
    for z in range(iter):
        rows = len(data1[e][z])
        pareto_unsort = np.array(data1[e][z])
        pareto_sort = pareto_unsort[pareto_unsort[:,1].argsort()] # rank the solutions of iteration z in a descending order according to their communication rate
        hypervolume  = 0
        for i in range(rows):
                long = math.sqrt(math.pow(pareto_sort[i][0] - nadir_x,2) + math.pow(pareto_sort[i][1] - pareto_sort[i][1],2))
                larg = math.sqrt(math.pow(pareto_sort[i][0] - pareto_sort[i][0],2) + math.pow(pareto_sort[i][1] - nadir_y,2))
                hypervolume = hypervolume + (long * larg)
        y = np.append(y,hypervolume)
    y  = np.delete(y,0,0) # delete the first useless zero
    print(y)
    plt.show()
    plt.savefig("hypervolume.esp",format="eps")
    a = plt.plot(x,y)