from __future__ import print_function
from scoop import futures
import json
from fl_gpu import JSDEvaluate 


if __name__ == "__main__":
    ev = JSDEvaluate()

    myseeds = [1, 62011,80177,97213,109567,181327103,117797,122393,130841,137803,141223,144961,149749,155657,159193,163679,167801,173137,184649,189407,198529,204047,208843,214789,221077,227219,233297,200604289,251623,256423,263387, 179426549, 1300609]
#[62011,80177,97213,109567,181327103,117797,122393,130841,137803,141223,144961,149749,155657,159193,163679,167801,173137,184649,189407,198529,204047,208843,214789,221077,227219,233297,200604289,251623,256423,263387, 179426549, 1300609]

    chosen_topology = "CONV"
    n_layers = 12
    n_clients = 4
    local_steps = 1
    
    n_seeds = 30
    '''
    solution = [32] * n_layers + [n_clients] + [local_steps] + [0] * n_layers
    results = list(futures.map(ev.evaluate, [solution] * n_seeds, [chosen_topology] * n_seeds, myseeds[0:n_seeds]))   #  [(solution, chosen_topology, s) for s in myseeds]))
    with open('./full_communication_{0}_{1}.json'.format(chosen_topology, solution), 'w') as json_file:
        json.dump(str(results), json_file,indent=True)  

    solution = [2, 28, 8, 12, 1, 27, 23, 32, 12, 13, 1, 15, 3, 44, 17, 18, 13, 24, 0, 38, 4, 16, 5, 37, 44, 10]
    results = list(futures.map(ev.evaluate, [solution] * n_seeds, [chosen_topology] * n_seeds, myseeds[0:n_seeds]))
    with open('./best_09794_00116_{0}_{1}.json'.format(chosen_topology, solution), 'w') as json_file:
        json.dump(str(results), json_file,indent=True)  
    '''
    solution = [10, 14, 23, 16, 15, 17, 12, 14, 13, 13, 1, 17, 4, 130, 24, 23, 2, 21, 5, 40, 37, 25, 22, 33, 10, 14]
    results = list(futures.map(ev.evaluate, [solution] * n_seeds, [chosen_topology] * n_seeds, myseeds[0:n_seeds]))
    with open('./middle_09736_00051_{0}_{1}.json'.format(chosen_topology, solution), 'w') as json_file:
        json.dump(str(results), json_file,indent=True)  

    '''
    solution = [3,20,14,25,2,21,9,21,9,12,1,7,3,14,27,17,3,1,6,33,44,8,10,1,32,9]
    results = list(futures.map(ev.evaluate, [solution] * n_seeds, [chosen_topology] * n_seeds, myseeds[0:n_seeds]))
    with open('./best_09836_01855_{0}_{1}.json'.format(chosen_topology, solution), 'w') as json_file:
        json.dump(str(results), json_file,indent=True)  

    solution = [5,25,7,12,2,9,7,11,8,3,2,25,2,42,34,6,6,12,12,3,4,37,50,44,29,3]
    results = list(futures.map(ev.evaluate, [solution] * n_seeds, [chosen_topology] * n_seeds, myseeds[0:n_seeds]))
    with open('./middle_09781_00643_{0}_{1}.json'.format(chosen_topology, solution), 'w') as json_file:
        json.dump(str(results), json_file,indent=True)  
    '''
    '''
    filename = "full_communication"
    solution = [32] * n_layers + [n_clients] + [local_steps] + [0] * n_layers
    futures.map(myeval, [(filename, ev, solution, chosen_topology, s) for s in myseeds])

    filename = "best_09836_01855"
    solution = solution = [3,20,14,25,2,21,9,21,9,12,1,7,3,14,27,3,1,6,33,44,8,10,1,32,9]
    futures.map(myeval, [(filename, ev, solution, chosen_topology, s) for s in myseeds])

    filename = "middle_09781_00643"
    solution = [5,25,7,12,2,9,7,11,8,3,2,25,2,42,34,6,6,12,12,3,4,37,50,44,29,3]
    futures.map(myeval, [(filename, ev, solution, chosen_topology, s) for s in myseeds])

    results = []
    solution = [32] * n_layers + [n_clients] + [local_steps] + [0] * n_layers
    for i in range(30):
      results.append(ev.evaluate(solution, chosen_topology, myseeds[i]))
    with open('./full_communication_{0}_{1}_{2}_{3}_{4}.json'.format(myseeds[i], chosen_toplogy, solution), 'w') as json_file:
        json.dump(solutions, json_file,indent=True)
    
    results = []
    solution = [3,20,14,25,2,21,9,21,9,12,1,7,3,14,27,3,1,6,33,44,8,10,1,32,9]
    for i in range(30):
      results.append(ev.evaluate(solution, chosen_topology, myseeds[i]))
    with open('./best_communication_09836_01855_{0}_{1}_{2}_{3}_{4}.json'.format(myseeds[i], chosen_toplogy, solution), 'w') as json_file:
        json.dump(solutions, json_file,indent=True)

    results = []

    solution = [5,25,7,12,2,9,7,11,8,3,2,25,2,42,34,6,6,12,12,3,4,37,50,44,29,3]
    for i in range(30):
      results.append(ev.evaluate(solution, chosen_topology, myseeds[i]))
    with open('./middle_communication_09781_00643_{0}_{1}_{2}_{3}_{4}.json'.format(myseeds[i], chosen_toplogy, solution), 'w') as json_file:
        json.dump(solutions, json_file,indent=True)
    '''
