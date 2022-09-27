#!/usr/bin/env python3

__appname__ = '[App_name_here]'
__author__ = 'Pablo Lechon (plechon@ucm.es)'
__version__ = '0.0.1'

## IMPORTS ##

import sys
import numpy as np
import pandas as pd
from essential_tools import *

## CONSTANTS ##


## FUNCTIONS ##

def simulate_data(n, A, rho, design_matrix):
    '''
    Generate simulated observations with a GLV
    
    Parameters:
        n (int): Number of species (consumers+resources if CR)
        A (nxn array): Matrix of interactions
        rho (nx1 float): Species growth rates
        design_matrix (kxn array): Matrix of presence absence of each species
                                   in each experiment
    '''
    #create community
    glv_community = Community(np.ones(n), GLV, A=A, rho=rho)
    #preallocate storage for output data
    k = len(design_matrix)
    data = np.zeros((k, n))
    for i in range(k):
        #get indices of removed species (note the +m in the end)
        rem_spp_ind = np.where(design_matrix[i,:]==0)[0]
        present = np.where(design_matrix[i,:]==1)[0]
        #remove spp
        subcomm = glv_community.remove_spp(rem_spp_ind)
        subcomm.assembly()
        data[i,present] = subcomm.n
        if any(data[i, present] == 0):
            return np.array([False], dtype=bool)
    return data

def main(argv):
    '''Main function'''
    #Set parameters
    n, m = (3, 3)
    feasible = False
    #sample growth rates and interactions
    while not feasible:
        d = np.repeat(np.random.uniform(0, 1), n)
        r = 1+max(d)+np.random.uniform(0, 1, m)
        C = np.random.uniform(0, 1, size=(m,n))
        #full GLV system
        A = np.block([[np.identity(m), C], [-C.T, np.zeros((n, n))]])
        rho = np.concatenate((r, -d))
        #Set experimental design matrix
        res_mat = np.ones((m, m))
        spp_mat = np.identity(n)
        design_mat = np.hstack((res_mat, spp_mat))
        #generate simulated data
        data = simulate_data(n+m, A, rho, design_mat)
        if data.any(): 
            feasible = True
    import ipdb; ipdb.set_trace(context = 20)
    #2. Propose a C
    #3. Build corresponding A
    #4. Compute z with the proposed A
    #5. Compute P^T = Az^T
    #6. Find an updated A^-1 by numerically minimizing ||\sim z^T - A^-1P^T||
    #7. Invert A^-1 to recover A.
    #8. Repeat 3-7 until convergence

    return 0

## CODE ##

if (__name__ == '__main__'):
    status = main(sys.argv)
    sys.exit(status)
     

