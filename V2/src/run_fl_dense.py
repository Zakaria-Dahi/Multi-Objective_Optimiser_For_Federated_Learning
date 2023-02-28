from __future__ import print_function
from scoop import futures
import json
from fl_gpu import JSDEvaluate 


if __name__ == "__main__":
    ev = JSDEvaluate()

    myseeds = [1, 62011,80177,97213,109567,181327103,117797,122393,130841,137803,141223,144961,149749,155657,159193,163679,167801,173137,184649,189407,198529,204047,208843,214789,221077,227219,233297,200604289,251623,256423,263387, 179426549, 1300609]
#[62011,80177,97213,109567,181327103,117797,122393,130841,137803,141223,144961,149749,155657,159193,163679,167801,173137,184649,189407,198529,204047,208843,214789,221077,227219,233297,200604289,251623,256423,263387, 179426549, 1300609]

    chosen_topology = "DENSE"#"CONV"
    n_layers = 4 #12
    n_clients = 4
    local_steps = 1

    n_seeds = 30

    solution = [32] * n_layers + [n_clients] + [local_steps] + [0] * n_layers
    results = list(futures.map(ev.evaluate, [solution] * n_seeds, [chosen_topology] * n_seeds, myseeds[0:n_seeds]))
    print(results)
    with open('./full_communication_{0}_{1}.json'.format(chosen_topology, solution), 'w') as json_file:
        json.dump(str(results), json_file,indent=True)  


    solution = [9 , 7 , 1 , 17 , 4 , 6 , 6 , 35 , 31 , 11] #[6,16,1,19,4,26,8,37,49,50]
    results = list(futures.map(ev.evaluate, [solution] * n_seeds, [chosen_topology] * n_seeds, myseeds[0:n_seeds]))
    print(results)
    with open('./best_09429_0105_{0}_{1}.json'.format(chosen_topology, solution), 'w') as json_file:
        json.dump(str(results), json_file,indent=True)  

    solution = [4 , 14 , 5 , 13 , 3 , 904 , 9 , 42 , 16 , 14]
    results = list(futures.map(ev.evaluate, [solution] * n_seeds, [chosen_topology] * n_seeds, myseeds[0:n_seeds]))
    print(results)
    with open('./middle_09367_0005_{0}_{1}.json'.format(chosen_topology, solution), 'w') as json_file:
        json.dump(str(results), json_file,indent=True)  


    '''
    solution = [15,7,1,11,4,6,6,35,41,35] #[6,16,1,19,4,26,8,37,49,50]
    results = list(futures.map(ev.evaluate, [solution] * n_seeds, [chosen_topology] * n_seeds, myseeds[0:n_seeds]))
    print(results)
    with open('./best_09417_01194_{0}_{1}.json'.format(chosen_topology, solution), 'w') as json_file:
        json.dump(str(results), json_file,indent=True)  

    solution = [31,10,1,11,4,32,1,46,40,48]
    results = list(futures.map(ev.evaluate, [solution] * n_seeds, [chosen_topology] * n_seeds, myseeds[0:n_seeds]))
    print(results)
    with open('./middle_09408_003_{0}_{1}.json'.format(chosen_topology, solution), 'w') as json_file:
        json.dump(str(results), json_file,indent=True)  
    '''
    '''
    filename = "full_communication"
    solution = [32] * n_layers + [n_clients] + [local_steps] + [0] * n_layers
    futures.map(myeval, [(filename, ev, solution, chosen_topology, s) for s in myseeds])

    filename = "best_09424_01678"
    solution = [6,16,1,19,4,26,8,37,49,50]
    futures.map(myeval, [(filename, ev, solution, chosen_topology, s) for s in myseeds])

    filename = "middle_09359_00285"
    solution = [4,2,13,8,3,911,43,50,7,32]
    futures.map(myeval, [(filename, ev, solution, chosen_topology, s) for s in myseeds])
    #futures


    solution = [32] * n_layers + [n_clients] + [local_steps] + [0] * n_layers
    results = []
    for i in range(30):
      results.append(ev.evaluate(solution, chosen_topology, myseeds[i]))
    with open('./full_communication_{0}_{1}_{2}_{3}_{4}.json'.format(myseeds[i], chosen_toplogy, solution), 'w') as json_file:
        json.dump(solutions, json_file,indent=True)
    
    results = []
    solution = [6,16,1,19,4,26,8,37,49,50]
    for i in range(30):
      results.append(ev.evaluate(solution, chosen_topology, myseeds[i]))
    with open('./best_communication_09424_01678_{0}_{1}_{2}_{3}_{4}.json'.format(myseeds[i], chosen_toplogy, solution), 'w') as json_file:
        json.dump(solutions, json_file,indent=True)

    results = []

    solution = [4,2,13,8,3,911,43,50,7,32]
    for i in range(30):
      results.append(ev.evaluate(solution, chosen_topology, myseeds[i]))
    with open('./middle_communication_09359_00285_{0}_{1}_{2}_{3}_{4}.json'.format(myseeds[i], chosen_toplogy, solution), 'w') as json_file:
        json.dump(solutions, json_file,indent=True)
    '''
  
