#!/usr/bin/env python3

__appname__ = '[App_name_here]'
__author__ = 'Pablo Lechon (plechon@ucm.es)'
__version__ = '0.0.1'

## IMPORTS ##

import sys
from essential_tools import *
from scipy.stats import beta, dirichlet
import numpy as np

## CONSTANTS ##


## FUNCTIONS ##

def get_C(n_resources, n_consumers, n_preferences):
    '''
    Construct matrix C (n_resources x n_consumers)

    n_resources (int): Number or resources
    n_consumers (int): Number of consumers
    n_pref (1xn_consumers): Number of preferences of each consumer
    '''
    #initialize C
    C = np.zeros(shape=(n_resources, n_consumers))
    for i in range(n_consumers):
        shuffle_idx = np.random.permutation(np.arange(0, n_resources))
        resource_id = shuffle_idx[:n_preferences[i]]
        #assign rates to selected resources
        C[resource_id, i] = np.random.dirichlet(np.ones(n_preferences[i]))
    return C

def main(argv):
    '''Main function'''
    #set resources parameters 
    m = 10
    r = np.random.uniform(0.5, 1, size=m)
    #set species parameters
    n = 10
    d = np.random.uniform(0, 0.5, size=n)
    n_pref = np.random.random_integers(1, m, size=n)
    C = get_C(m, n, n_pref)
    #construct GLV parameters
    I = np.identity(m)
    O = np.zeros(shape=(n, n))
    A = np.block([[-I, -C], [C.T, O]])
    #resources first, consumers second
    rho = np.hstack((r, -d))
    #initialize community object
    comm = Community(np.ones(m+n), GLV, A=A, r=rho)
    comm.assembly(t_dynamics = True)
    comm.plotter()
    sub_comm = comm.remove_spp(np.array([0]), hard_remove = False)
    sub_comm.assembly(t_dynamics = True)
    import ipdb; ipdb.set_trace(context = 20)
    return 0

## CODE ##

if (__name__ == '__main__'):
    status = main(sys.argv)
    sys.exit(status)
     

