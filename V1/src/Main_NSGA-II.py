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

import array
import random
import json

import numpy

from math import sqrt

from deap import algorithms
from deap import base
from deap import benchmarks
from deap.benchmarks.tools import diversity, convergence, hypervolume
from deap import creator
from deap import tools
from initialize import initialize
from FL_OF import FL_OF
from bounding import bounding
import matplotlib.pyplot as plt

# Define the parameters to optimise
lb1 = 1;
ub1 = 32;
lay = 4; # number of neural network layers
lb2 = 1;
ub2 = 4; 
esc = 1 # 1 dimension to represent number of slaves participating in the communication
lb3 = 1;
ub3 = 1000;
ts = 1 # 1 dimension to represent the number training steps
lb4 = 0;
ub4 = 50;
thre =  lay # 1 dimension to establish the threshold
BOUND_LOW = min([lb1,lb2,lb3,lb4])
BOUND_UP = max([ub1,ub2,ub3,ub4])


"""
# solution is organised as
    precision for each layer
    number of slaves to communicate
    number of training steps
    threshold for each layer
"""   

# size of the solution
NDIM = lay + esc + ts + thre; 

creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()

def uniform(low, up, size=None):
    try:
        return [random.uniform(a, b) for a, b in zip(low, up)]
    except TypeError:
        return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]


toolbox.register("attr_float", initialize, lb1,ub1,lay,lb2,ub2,esc,lb3,ub3,ts,lb4,ub4,thre)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", FL_OF)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM)
toolbox.register("select", tools.selNSGA2)

def main(seed=None):
    random.seed(seed)

    NGEN = 200 # the number of iterations
    MU = 400 # size of the population
    CXPB = 0.9

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean, axis=0)
    stats.register("std", numpy.std, axis=0)
    stats.register("min", numpy.min, axis=0)
    stats.register("max", numpy.max, axis=0)
    
    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"
    
    pop = toolbox.population(n=MU)
    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit # this is where the idnividuals are evaluated

    # This is just to assign the crowding distance to the individuals
    # no actual selection is done
    pop = toolbox.select(pop, len(pop))
    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(invalid_ind), **record)
    # print(logbook.stream)

    # Begin the generational process
    for gen in range(1, NGEN):
        # Vary the population
        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]
        
        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() <= CXPB:
                toolbox.mate(ind1, ind2)
                """
                apply the bouding on the resulting individuals
                """
                ind1 = bounding(ind1,lb1,ub1,lay,lb2,ub2,esc,lb3,ub3,ts,lb4,ub4,thre)
                ind2 = bounding(ind2,lb1,ub1,lay,lb2,ub2,esc,lb3,ub3,ts,lb4,ub4,thre)
            toolbox.mutate(ind1)
            toolbox.mutate(ind2)
            """
            apply the bouding on the resulting individuals
            """
            ind1 = bounding(ind1,lb1,ub1,lay,lb2,ub2,esc,lb3,ub3,ts,lb4,ub4,thre)
            ind2 = bounding(ind2,lb1,ub1,lay,lb2,ub2,esc,lb3,ub3,ts,lb4,ub4,thre)
            del ind1.fitness.values, ind2.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Select the next generation population
        pop = toolbox.select(pop + offspring, MU)
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        #print(logbook.stream)

    print("Final population hypervolume is %f" % hypervolume(pop, [11.0, 11.0]))

    return pop, logbook
        
if __name__ == "__main__":
    """
    with open("pareto_front/zdt1_front.json") as optimal_front_data:
        optimal_front = json.load(optimal_front_data)
    # Use 500 of the 1000 points in the json file
    optimal_front = sorted(optimal_front[i] for i in range(0, len(optimal_front), 2))
    """
    pop, stats = main()
    pop.sort(key=lambda x: x.fitness.values)
    """
    print(stats)
    print("Convergence: ", convergence(pop, optimal_front))
    print("Diversity: ", diversity(pop, optimal_front[0], optimal_front[-1]))
     """ 
    front = numpy.array([ind.fitness.values for ind in pop])
    """
    optimal_front = numpy.array(optimal_front)
    plt.scatter(optimal_front[:,0], optimal_front[:,1], c="r")
    """
    plt.scatter(front[:,0], front[:,1], c="b")
    plt.axis("tight")
    plt.show()