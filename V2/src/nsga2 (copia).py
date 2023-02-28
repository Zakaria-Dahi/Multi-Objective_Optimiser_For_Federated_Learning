#    This file is part of DEAP.
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.


#https://stackoverflow.com/questions/53732589/how-to-set-upper-and-lower-bounds-to-a-gene-in-an-individual-in-deap

import sys
import array
import random
import json

import numpy as np

import collections
import os
import time
start_time = time.time()

from math import sqrt

    
import matplotlib.pyplot as plt


from deap import algorithms
from deap import base
from deap import benchmarks
from deap.benchmarks.tools import diversity, convergence, hypervolume
from deap import creator
from deap import tools

import logging
logging.basicConfig(
    #filename='HISTORYlistener.log',
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)



from FL_OF import FL_OF
from fl_gpu import JSDEvaluate 




class ProblemParameters():
  def __init__(self,  chosen_topology, nlayers, ga_seed):
    # Problem definition
    # Define the parameters to optimise
    self.lb1 = 1 #1
    self.ub1 = 32
    self.lay = nlayers #4 #12 #4#12 # number of neural network layers
    self.lb2 = 1
    self.ub2 = 4 
    self.esc = 1 # 1 dimension to represent number of slaves participating in the communication
    self.lb3 = 1
    self.ub3 = 1000
    self.ts = 1 # 1 dimension to represent the number training steps
    self.lb4 = 0
    self.ub4 = 50
    self.thre =  nlayers # 1 dimension to establish the threshold

    self.CHOSEN_TOPOLOGY = chosen_topology #"DENSE" #"CONV"
    self.N_ITERATIONS = 302 #3000
    self.POP_SIZE = 100
    self.INDIVIDUAL_SIZE = nlayers * 2 + 2
    self.CXPB = 0.9
    self.MUTATEPB = 1.0 / self.INDIVIDUAL_SIZE #2.0 / self.INDIVIDUAL_SIZE

    self.ga_seed = ga_seed









def max_communication_individual(pp): ##pp = problem_params  
    size = pp.lay * 2 + 2
    solution = [0] * size
    n_layers = pp.lay
    idx_quantization = 0
    idx_slaves = idx_quantization + n_layers
    idx_steps = idx_slaves + 1
    idx_threshold = idx_steps + 1

    idx = idx_quantization
    while idx < idx_quantization + n_layers:
      solution[idx] = pp.ub1
      idx += 1
    
    solution[idx_slaves] = pp.ub2
    solution[idx_steps] = pp.ub3


    idx = idx_threshold
    while idx < idx_threshold + n_layers:
      solution[idx] = pp.ub4
      idx += 1

    return solution


def myinitialization(pp):
    size = pp.lay * 2 + 2
    solution = [0] * size
    n_layers = pp.lay
    idx_quantization = 0
    idx_slaves = idx_quantization + n_layers
    idx_steps = idx_slaves + 1
    idx_threshold = idx_steps + 1

    idx = idx_quantization
    while idx < idx_quantization + n_layers:
      solution[idx] = random.randint(pp.lb1, pp.ub1)
      idx += 1
    
    solution[idx_slaves] = random.randint(pp.lb2, pp.ub2)
    solution[idx_steps] = random.randint(pp.lb3, pp.ub3)


    idx = idx_threshold
    while idx < idx_threshold + n_layers:
      solution[idx] = random.randint(pp.lb4, pp.ub4)
      idx += 1

    return solution


def save_stats(pop, stats, num_gen, pp):
    pop.sort(key=lambda x: x.fitness.values)
    
    logging.info(stats)
    #logging.info("Convergence: ", convergence(pop, optimal_front))
    #logging.info("Diversity: ", diversity(pop, optimal_front[0], optimal_front[-1]))

    '''
    front = np.array([ind.fitness.values for ind in pop])
    optimal_front = np.array(optimal_front)
    plt.scatter(optimal_front[:,0], optimal_front[:,1], c="r")
    plt.scatter(front[:,0], front[:,1], c="b")
    plt.axis("tight")
    plt.show()
    '''


    path_folder = './stats_{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}_{8}_{9}_{10}_{11}_{12}_{13}_{14}_{15}_{16}_{17}'.format(pp.ga_seed, pp.N_ITERATIONS, pp.CHOSEN_TOPOLOGY, pp.POP_SIZE, pp.CXPB, pp.MUTATEPB, pp.lb1, pp.ub1, pp.lay, pp.lb2, pp.ub2, pp.esc, pp.lb3, pp.ub3, pp.ts, pp.lb4, pp.ub4, pp.thre)
    os.makedirs(path_folder, exist_ok=True)

    # Plot 
    plt.title("Pareto front")
    plt.xlabel("Accuracy")
    plt.ylabel("Communications")
    #plt.plot(front[:,0],front[:,1],"r--") #plt.scatter(front[:,0], front[:,1], c="b")
    front = np.array([ind.fitness.values for ind in pop])  
    plt.scatter(front[:,0], front[:,1], c="b")  
    plt.grid(True)
    plt.axis("tight")
    #plt.show()
    plt.savefig(path_folder + '/pareto_initt_{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}_{8}_{9}_{10}_{11}_{12}_{13}_{14}_{15}_{16}_{17}_{18}.pdf'.format(pp.ga_seed, num_gen, pp.N_ITERATIONS, pp.CHOSEN_TOPOLOGY, pp.POP_SIZE, pp.CXPB, pp.MUTATEPB, pp.lb1, pp.ub1, pp.lay, pp.lb2, pp.ub2, pp.esc, pp.lb3, pp.ub3, pp.ts, pp.lb4, pp.ub4, pp.thre), dpi=600)

    solutions=tree()
    # Evaluate the running time
    gen, evals, std, minn, avg, maxx = stats.select("gen", "evals", "std", "min", "avg", "max")
    t=time.time() - start_time

    solutions['population'] = str(pop)
    pop_fitness = []
    for i in range(len(pop)):
        pop_fitness.append(pop[i].fitness.values)
    solutions['population_fitness'] = str(pop_fitness)
    #solutions['convergence'] = convergence(pop, optimal_front)
    #solutions['diversity'] = diversity(pop, optimal_front[0], optimal_front[-1])
    solutions['time']=t

    ncores = multiprocessing.cpu_count()
    solutions['cores']= multiprocessing.cpu_count() #1#int(os.environ["SLURM_NTASKS"])
    solutions['stats'] = str(stats)
    solutions['hypervolume'] = hypervolume(pop)
    solutions['gen'] = str(gen)
    solutions['evals'] = str(evals)
    solutions['std'] = str(std)
    solutions['min'] = str(minn)
    solutions['avg'] = str(avg)
    solutions['max'] = str(maxx)

    random_state = random.getstate()
    nprandom_state = np.random.get_state()
    solutions['random_state'] = str(random_state)
    solutions['nprandom_state'] = str(nprandom_state)
    
    with open(path_folder + '/stats_initt_{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}_{8}_{9}_{10}_{11}_{12}_{13}_{14}_{15}_{16}_{17}_{18}.json'.format(pp.ga_seed, num_gen, pp.N_ITERATIONS, pp.CHOSEN_TOPOLOGY, pp.POP_SIZE, pp.CXPB, pp.MUTATEPB, pp.lb1, pp.ub1, pp.lay, pp.lb2, pp.ub2, pp.esc, pp.lb3, pp.ub3, pp.ts, pp.lb4, pp.ub4, pp.thre), 'w') as json_file:
        json.dump(solutions, json_file,indent=True)

from scoop import futures
import multiprocessing #for counting number of cores





def tree():
    ''' 
        Recursive dictionnary with defaultdict 
    '''
    return collections.defaultdict(tree)



toolbox = base.Toolbox()
toolbox.register("map", futures.map)
creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))
creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMulti)


def main(pp):
    global toolbox
    random.seed(pp.ga_seed)
    np.random.seed(pp.ga_seed) 

    
    ev = JSDEvaluate()



    toolbox.register("attr_int", myinitialization, pp)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_int)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    lower_bounds = [pp.lb1] * pp.lay + [pp.lb2] + [pp.lb3] + [pp.lb4] * pp.lay
    upper_bounds = [pp.ub1] * pp.lay + [pp.ub2] + [pp.ub3] + [pp.ub4] * pp.lay
    toolbox.register("evaluate", FL_OF, evaluator = ev, chosen_topology = pp.CHOSEN_TOPOLOGY)

    #toolbox.register("mate", tools.cxSimulatedBinaryBounded, eta=20.0, low = lower_bounds, up = upper_bounds)
    toolbox.register("mate", tools.cxOnePoint)
    #toolbox.register("mutate", tools.mutUniformInt, low = lower_bounds, up = upper_bounds, indpb=MUTATEPB)
    #toolbox.register("mutate", tools.mutPolynomialBounded, low=lower_bounds, up=upper_bounds, eta=20.0, indpb=pp.MUTATEPB)
    toolbox.register("mutate", tools.mutUniformInt, low=lower_bounds, up=upper_bounds, indpb=pp.MUTATEPB)
    toolbox.register("select", tools.selNSGA2)

    ######################################


    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)
    
    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"
    
    pop = toolbox.population(n = pp.POP_SIZE)
    pop[0] = creator.Individual(max_communication_individual(pp)) # add default individual
    

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # This is just to assign the crowding distance to the individuals
    # no actual selection is done
    pop = toolbox.select(pop, len(pop))
    
    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(invalid_ind), **record)
    logging.info(logbook.stream)

    # Begin the generational process
    for gen in range(1, pp.N_ITERATIONS):
        print("\n")
        logging.info("ITERATION = " + str(gen))
        # Vary the population
        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]
        
        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() <= pp.CXPB:
                toolbox.mate(ind1, ind2)
            
            toolbox.mutate(ind1)
            toolbox.mutate(ind2)
         
            del ind1.fitness.values, ind2.fitness.values
        
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Select the next generation population
        pop = toolbox.select(pop + offspring, pp.POP_SIZE)
        record = stats.compile(pop)
        logbook.record(gen = gen, evals = len(invalid_ind), **record)
        logging.info(logbook.stream)

        save_stats(pop, logbook, gen, pp)

    #logging.info("Final population hypervolume is %f" % hypervolume(pop, [11.0, 11.0]))

    return pop, logbook
        
if __name__ == "__main__":
    #with open("pareto_front/zdt1_front.json") as optimal_front_data:
    #    optimal_front = json.load(optimal_front_data)
    # Use 500 of the 1000 points in the json file
    #optimal_front = sorted(optimal_front[i] for i in range(0, len(optimal_front), 2))
    


    print("sys.argv  =  " + str(sys.argv))

    ga_seed = int(sys.argv[1]) #1
    chosen_topology = sys.argv[2]
    nlayers = int(sys.argv[3])
    print("ga_seed = " + str(ga_seed))
    print("CHOSEN_TOPOLOGY = " + str(chosen_topology))
    print("nlayers = " + str(nlayers))
    myseeds = [62011,80177,97213,109567,181327103,117797,122393,130841,137803,141223,144961,149749,155657,159193,163679,167801,173137,184649,189407,198529,204047,208843,214789,221077,227219,233297,200604289,251623,256423,263387, 179426549, 1300609]
    #myseeds = [62011,80177,109567,117797,122393,130841,137803,141223,144961,149749,159193,163679,167801,173137,184649,189407,198529,204047,208843,221077,227219,200604289,251623, 179426549, 1300609]
    #myseeds = [62011,109567,117797,122393,130841,137803,141223,144961,159193,163679,167801,184649,189407,198529,208843,221077,227219,200604289, 1300609]    
    #myseeds = [62011,109567,117797,122393,130841,141223,144961,159193,163679,167801,184649,189407,198529,208843,221077,227219, 1300609]
    #myseeds = [62011,163679, 1300609]
    ga_seed = myseeds[ga_seed]
    pp = ProblemParameters(chosen_topology, nlayers, ga_seed) ##pp = problem_params
    pop, stats = main(pp)

    save_stats(pop, stats, pp.NN_ITERATIONS, pp)

    

'''
6 - 122393
9 - 141223
16 - 173137
20 - 204047
22 - 214789
25 - 233297
26 - 200604289
27- 251623
29 - 263387
30 - 179426549



3 - 109567
5- 117797
7- 130841
8- 137803
10 - 144961
11 - 149749
12 - 155657

1- 80177
2- 97213
4- 181327103

14- 163679
15- 167801

17
19
23
28
'''

