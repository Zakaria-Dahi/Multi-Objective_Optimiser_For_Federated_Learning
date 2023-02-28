#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 18:46:22 2021

@author: user
"""

def bounding(indiv,lb1,ub1,lay,lb2,ub2,esc,lb3,ub3,ts,lb4,ub4,thre):
    for index in range (len(indiv)):
        indiv[index] = round(indiv[index])
    # apply upper and lower bound on the weights precisions
    index = 0;        
    while index < lay:
        indiv[index] = round(indiv[index])
        if indiv[index]<lb1:
            indiv[index] = lb1;
        if indiv[index]<ub1:
            indiv[index] = ub1 ;      
        index += index+1;
    # apply upper and lower bound on the number of slaves
    while index < (lay+esc):
        indiv[index] = round(indiv[index])
        if indiv[index]<lb2:
            indiv[index] = lb2
        if indiv[index]<ub2:
            indiv[index] = ub2
        index = index+1;
    # apply upper and lower bound on the number of training steps
    while index < (lay+esc+ts):
        indiv[index] = round(indiv[index])
        if indiv[index]<lb3:
            indiv[index] = lb3
        if indiv[index]<ub3:
            indiv[index] = ub3
        index = index+1;
    # apply upper and lower bound on the number of training steps
    while index < (lay+esc+thre):
        indiv[index] = round(indiv[index])
        if indiv[index]<lb4:
            indiv[index] = lb4
        if indiv[index]<ub4:
            indiv[index] = ub4
        index = index+1;
    return indiv;