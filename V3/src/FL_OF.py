#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 18:05:55 2021

@author: user
"""

import random

def FL_OF(solution, evaluator, chosen_topology):
    
    #print(solution)
    #print(evaluator)
    return evaluator.evaluate(solution, chosen_topology)
    '''
    # f1 stocks the communication cost after executing our Federated-learning ANN
    f1 = sum(solution)*random.uniform(0,1)
    # f2 stocks the accuracy of our Federated-learning ANN
    f2 = (-1)*sum(solution)*random.uniform(0,1)
    return f1, f2;
    '''
