# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 22:11:31 2022
Author: Kylel Scott
Course: CSE 598: Bio-Inspired AI and Optimization
Date: 2/10/21
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd

"""
Assignment:
    Genetic Algorithmn implementation:
        - optimizaiton function: f = x*sin(10*x*x) +1
        - x = [-.5, 1]
"""


NUM_GENES = 5
POP_SIZE = 8
x = np.arange(-.5,1,.01)

"""
Helper Functions
"""

def function(x):
    """
    Used to assess x values for f(x) = x*sin(10*x*x) +1
    """
    return (x*np.sin(10*math.pi*x)+1)


def create_parent(num_genes):
    """
    Function used to create a parent individual. Samples random numerical 
    values for genotype encoding. 

    Parameters
    ----------
    num_genes : number of genes for the parent to have
    
    Returns
    -------
    p : encoding parent individual
    """
    p = np.empty((1,num_genes))
    p[0][0] = np.random.uniform(-1,1)
    for i in range(1,num_genes):
        p[0][i] = np.random.randint(0,9) #fix to change last number as well
    return p


def decode_parent(parent):
    """
    Function used to decode encoded parent vector into a value for function eval.

    Parameters
    ----------
    parent : encoded parent vector
    
    Returns
    -------
    evaluation of parent on f(x)
    """
    sign = 1
    if parent[0][0] < 0: 
        sign = -1
    number = parent[0][1]/10 + parent[0][2]/100 + parent[0][3]/1000 + parent[0][4]/10000
    return sign * number
    

def fit_parent(parent):
    """
    Function used to fit parent to decode and evaluate parent on f(x)

    Parameters
    ----------
    parent : encoded parent vector

    Returns
    -------
    evaluation of f(x) where x= parent_vector
    """
    parent = decode_parent(parent)
    penalty = 0
    if parent < .5:
        penalty = -5
    return function(parent) + penalty



"""
Functions
"""



def generate_population(num_genes, pop_size):
    """
    This fucntion generates a population of M x N size, where M is the number of individuals 
    and N is the number of genes per individual.    

    Parameters
    ----------
    num_genes : number of genes for individuals to have in population
    pop_size : size of population

    Returns
    -------
    population : population array
    """
    
    if pop_size % 2 != 0: #have to have even population size
        pop_size + 1
    population = np.empty((pop_size,num_genes))
    for i in range(0,pop_size):
        parent = create_parent(num_genes)
        population[i] = parent[0]
    return population


def fit_population(population, debug=True):
    """
    This fucntion fits a population and organizes the results in a dictionary for 
    calling

    Parameters
    ----------
    population : population array

    Returns
    -------
    fitness_chart : list containing fitness information
        DESCRIPTION.
    fitness_dict : dictionary object containing parent individual information
    """
    
    fitness_chart = []
    fitness_dict = {
        "parent": [],
        "decoded parent": [],
        "fitness_score": []}
    for parent in population: #for each parent, 1) decode 2) eval fitness 3) append to dict
        parent = [parent]
        fitness_score = fit_parent(parent)
        fitness_chart.append(fitness_score)
        fitness_dict["parent"].extend(parent)
        fitness_dict["decoded parent"].extend([(parent)])
        fitness_dict["fitness_score"].extend([fitness_score])
    fitness_dict = pd.DataFrame.from_dict(fitness_dict)
    return fitness_chart, fitness_dict


def selection_operator(fit_dict, sel_pressure=3):
    """
    Function to select candidates to proceed to next generation. The function runs N-many
    tournaments until a complete new generation population is conceived:
        - fit_dict:
            dictionary generated from fit_population that has important data on current generation
        - sel_pressure:
            hyperparameter that determines how many candidates enter tournament ea. time.
            The higher the #, the more participants, and the higher the SELECTIVE PRESSURE. 
            The lower the #, the least candidates, and the higher GENETIC DIVERSITY.
    """
    
    parents = np.empty((2, NUM_GENES))
    for num in range(0,2):
        population, fitnesses = fit_dict["parent"], fit_dict["fitness_score"]
        sel_idxs = []
        for i in range(sel_pressure): #chose n numbers of indexes to identify tournamnet players
            if sel_pressure > len(population):
                print("ERROR: SELECTIVE PRESSURE for SELECTION OPERATOR greater than number of candidates. This will create uneven distribution of parent participants in tournament")
                break
            idx = np.random.randint(0, len(population))
            sel_idxs.append(idx)
        players = [population[sel_idxs[i]] for i in range(sel_pressure)] #identify tournament players 
        p_fitnesses = [fitnesses[sel_idxs[j]] for j in range(sel_pressure)] #identify tournament players' fitnesses
        winner_fitness = np.max(p_fitnesses) #find winning fitness
        parent_idx = fit_dict["fitness_score"] == winner_fitness #find index of parent with winning fitness
        
        idx = population[parent_idx]
        idx = idx.keys()[0]
        parent = population[idx]
        try: 
            parents[num] = parent
        except KeyError: 
            print('KeyError: Key is not in parent dictionary.')
            continue
    return parents             


def one_point_crossover(parent1, parent2, cross_pt = 3):
    """
    Implementation of one point crossover

    Parameters
    ----------
    parent1 : parent vector
    parent2 : parent vector
    cross_pt : crossover point

    Returns
    -------
    child1 : offspring vector 1
    child2 : offspring vector 2
    """
    
    #should crossover include first element of parent vector?
    child1 = np.empty((parent1.shape))
    child2 = np.empty((parent1.shape))
    child1[0:cross_pt] = parent1[0:cross_pt]
    child1[cross_pt:] = parent2[cross_pt:]    
    child2[0:cross_pt] = parent2[0:cross_pt]
    child2[cross_pt:] = parent1[cross_pt:]
    return child1, child2

def mutation(child,P_m=.97): #Uniformly mutate gene with probability Pm
    """
    Uniform mutation implementation on gene with probability Pm

    Parameters
    ----------
    child : child vector
    P_m : Probability of mutation. The default is .97.

    Returns
    -------
    child : mutated child vector
    """
    
    for i in range(1,len(child)): 
        if np.random.uniform(0,1) > P_m:
            child[i] = np.random.randint(0,9)
        else:
            child[i] = child[i]
    return child


def gen_childs(parents):
    """
    Function used to generate child from a list of parent vectors

    Parameters
    ----------
    parents : list containing two parents

    Returns
    -------
    child1 : generated offspring
    child2 : generated offspring
    """
    
    child1, child2 = one_point_crossover(parents[0], parents[1]) #crossover btw parents
    child1 = mutation(child1) #mutation of child with Prob = P_m
    child2 = mutation(child2) #mutation of child with Prob = P_m
    return child1, child2
   

def gen_next_gen(population):
    """
    Function used to generate a new generation N+1 from generation N

    Parameters
    ----------
    population : population array

    Returns
    -------
    next_gen : new generation array
    """
    
    next_gen = np.empty((population.shape))
    children = []
    num_sessions = 3
    for i in range(num_sessions):
        fitness_chart, fitness_dict = fit_population(population) #fit population
        parents = selection_operator(fitness_dict) #select parent x 2
        child1, child2 = gen_childs(parents)
        children.append(child1)
        children.append(child2)
    next_gen = np.vstack([np.array(children), parents[0], parents[1]])

    print(parents, "\n")
    return next_gen


def plot_elites(elites, evolutions=0, save=True):
    decoded_elites_x = np.empty((POP_SIZE,))
    decoded_elites_y = np.empty((POP_SIZE,))
    
    #colors = ['r','g','b','m','y','cyan','magenta', 'k']
    for idx in range(0, len(elites)):
        decoded_elite = decode_parent([elites[idx]])
        decoded_elites_x[idx] = decoded_elite
        decoded_elites_y[idx] = function(decoded_elite)
    y = function(x)    
    plt.figure(2)
    plt.plot(x,y)
    plt.scatter(decoded_elites_x, decoded_elites_y, c='r')
    plt.title(f'Elite Plot for {evolutions} evolutions')
    if save:
        plt.savefig(f'figures/eli_plot_{evolutions}.png', bbox_inches='tight')
    plt.show()


def plot_function():
    y = function(x)
    plt.figure(1)
    plt.plot(x,y)
    plt.title('Optimization function')
    plt.show()
    
    
def evolve(init_population, num_evolutions, debug = True):
    """
    Function used to simulate evolution.

    Parameters
    ----------
    init_population : initial population
    num_of_generations : number of generations composed in evolution
    debug : Debug flag. The default is True.

    Returns
    -------
    elites : Optimal population after n = num_of_generations generations
    """
    
    population = init_population #initial population
    plot_function()

    for i in range(0, num_evolutions):
        print(f"Parents for Generation: {i}")
        new_population = gen_next_gen(population)
        population = new_population
        if debug:
            plot_elites(population, i, save=False)
    elites = population #final population
    plot_elites(elites, num_evolutions)
    return elites
    

init_population = generate_population(NUM_GENES, POP_SIZE)
elites = evolve(init_population, 100, True)
