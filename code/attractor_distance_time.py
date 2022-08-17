#!/usr/bin/env python3

__appname__ = '[App_name_here]'
__author__ = 'Pablo Lechon (plechon@ucm.es)'
__version__ = '0.0.1'

## IMPORTS ##

import sys
from essential_tools import *
from scipy.stats import beta, dirichlet
import numpy as np
import pandas as pd
import statsmodels.api as sm

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

def comm_function(C, abundances):
    x = abundances[:,np.newaxis]
    return (C@x).T[0]

def main(argv):
    '''Main function'''
    #set resources parameters 
    m = 10
    r = np.random.uniform(0.5, 1, size=m)
    #set species parameters
    n = 10
    d = np.random.uniform(0.3, 1, size=n)
    n_pref = np.random.random_integers(1, m, size=n)
    C = get_C(m, n, n_pref)
    #construct GLV parameters
    I = np.identity(m)
    O = np.zeros(shape=(n, n))
    A = np.block([[O, C.T], [-C, -I]])
    #resources first, consumers second
    rho = np.hstack((-d, r))
    #initialize community object
    comm = Community(np.ones(m+n), GLV, A=A, C=C, rho=rho, r=r, d=d)
    comm = comm.assembly(t_dynamics=True).delete_history()
    #get its function
    f = comm_function(comm.C, comm.n[:len(comm.r)])
    #which species are not extinct
    ind_pres = np.where(comm.presence[0:len(comm.r)] == 1)[0]
    #choose one
    ind_remove = np.random.choice(ind_pres)
    #remove history and one species
    sub_comm = comm.remove_spp(np.array([ind_remove])).delete_history()
    #assemble subcommunity
    sub_comm.assembly(t_dynamics=True)
    #get function by means of GLS
    gls_model = sm.GLS(f, sub_comm.C, sigma = np.identity(m))
    gls_results = gls_model.fit()
    ab_gls = gls_results.params
    f_opt = gls_results.predict()
    #compute ssr between functions along assembly trajectory
    t_points = sub_comm.t
    n_points = len(t_points)
    xi_t = np.zeros(n_points)
    for i in range(n_points):
        sp_abundances = sub_comm.abundances_t[0:len(sub_comm.r), i]
        f_k = comm_function(sub_comm.C, sp_abundances)
        res = f - f_k
        #get distance between f and f_k
        xi_t[i] = np.dot(res, res)
    #plot
    plt.plot(t_points, xi_t)
    plt.xscale('log')
    plt.show()
    return 0

## CODE ##

if (__name__ == '__main__'):
    status = main(sys.argv)
    sys.exit(status)
     

