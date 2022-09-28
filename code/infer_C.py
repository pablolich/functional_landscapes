#!/usr/bin/env python3

__appname__ = '[inference_C.py]'
__author__ = 'Pablo Lechon (plechon@uchicago.edu)'
__version__ = '0.0.1'

## DESCRIPTION ##
'''
This is code to check that you can infer matrix C from observing only the 
monocultures. In a GLV without cross-feeding.
'''
## IMPORTS ##

import sys
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from essential_tools import *
from scipy.optimize import minimize, Bounds
import progressbar

## CONSTANTS ##


## FUNCTIONS ##

def C2A(C):
    '''
    Cast matrix C into matrix A
    '''
    m, n = C.shape
    return np.block([[np.identity(m), C], [-C.T, np.zeros((n, n))]])

def A2C(A, m, n):
    '''
    Extract matrix C given matrix A
    '''
    return A[0:m, m:m+n]

def simulate_data(n, m, design_matrix):
    '''
    Simulate feasible experimental observation by assemblying subcommunities 
    with a GLV
    '''
    feasible = False
    #sample growth rates and interactions
    while not feasible:
        #draw random parameters
        d = np.repeat(np.random.uniform(0, 100), n)
        r = 1+max(d)+np.random.uniform(0, 1, m)
        C = np.random.uniform(0, 1, size=(m,n))
        #full GLV system
        A = C2A(C)
        rho = np.concatenate((r, -d))
        #create community
        glv_community = Community(np.ones(n+m), GLV, A=A, rho=rho)
        #preallocate storage for output data
        k = len(design_matrix)
        data = np.zeros((k, n+m))
        #assemble subcommunities
        for i in range(k):
            #get indices of removed and present species 
            rem_spp_ind = np.where(design_matrix[i,:]==0)[0]
            present = np.where(design_matrix[i,:]==1)[0]
            #remove spp and assemble subcommunity
            subcomm = glv_community.remove_spp(rem_spp_ind)
            subcomm.assembly()
            if any(subcomm.n == 0):
                print(k)
                break
            else: 
                #update row in matrix
                data[i,present] = subcomm.n
                if i==k-1: feasible = True
    return(data, A, rho)

def predict(A, rho, design_matrix):
    '''
    Predict abundances for each observation in the experimental design

    Parameters:
        A (nxn array): Matrix of interactions
        design_matrix (kxn array): Presence absence matrix for each species in
                                   each of the k experiemts.
    '''
    #number of species and experiments
    n = len(A)
    k = len(design_matrix)
    #preallocate matrix of predictions
    z_pred = np.zeros((k,n))
    for i in range(k):
        #indices of present and absent species
        delete_ind = np.where(design_matrix[i,:] == 0)[0]
        present = np.where(design_matrix[i,:] == 1)[0]
        #trim system
        A_i = np.delete(np.delete(A, delete_ind, axis=0), delete_ind, axis=1)
        #predict
        z_pred[i,present] = (np.linalg.inv(A_i)@rho[present, np.newaxis]).T
    return z_pred

def ssq(x, par, observations, design_matrix, n, m, C_var):
    '''
    Minimization goal function

    Parameters:
        x: vector of values to minimize (either C or rho)
        par: parameters that are fixed (either C or rho)
        observations: observed abundances
        design_matrix (kxn array): Presence absence matrix for each species in
                                   each of the k experiemts.
        n: Number of species
        m: Number of resources
    '''
    if C_var:
        C_vec = x
        C = C_vec.reshape(m, n)
        rho = par
    else:
        rho = x
        C_vec = par
        C = C_vec.reshape(m, n)
    #build matrix of interactions
    A = C2A(C)
    #predict abundances
    z_pred = predict(A, rho, design_matrix)
    #if any abundance is negative, multiply its value as penalization
    ind_neg = np.where(z_pred<0)
    z_pred[ind_neg] += 10*z_pred[ind_neg] 
    #calculate sum of squares
    ssq = np.sum((z_pred - observations)**2)
    return ssq 

def hill_climber(x, par, magnitude, n_steps, observations, design_matrix, 
                 n, m, C_var):
    '''
    Plain-vanilla hill climber algorithm that modifies parameters randomly
    '''
    #compute initial ssq
    SSQ = ssq(x, par, observations, design_matrix, n, m, C_var)
    for i in range(50):
        for i in range(n_steps):
            #sample perturbation
            p = np.random.normal(scale=magnitude, size=len(x))
            x_tmp = x*(1+p)
            SSQ_tmp = ssq(x_tmp, par, observations, design_matrix, n, m, C_var)
            if SSQ_tmp < SSQ:
                x = x_tmp
                SSQ = SSQ_tmp
                print('SSQ: ', SSQ)
        magnitude *= 0.95
    return x

def main(argv):
    '''Main function'''
    #Set parameters
    n, m = (7, 7)
    #create experimental design matrix
    res_mat = np.ones((m, m))
    spp_mat = np.identity(n)
    design_mat = np.hstack((res_mat, spp_mat))
    #generate data
    data, A, rho = simulate_data(n, m, design_mat)
    #set tolerance
    tol = 1e-10
    #propose a C
    C_cand = np.random.uniform(0, 1, size=(m, n))
    #propose a rho
    d = np.repeat(np.random.uniform(0, 1), n)
    r = 1+max(d)+np.random.uniform(0, 1, m)
    rho0 = np.concatenate((r, -d))
    #set bounds
    bounds_C = Bounds(0, 1)
    bounds_rho = Bounds(m*[0]+n*[-np.inf], m*[np.inf]+n*[0])
    #find best C for fixed rho
    C0 = hill_climber(C_cand, rho0, 1, 250, data, design_mat, n, m, True)
    for i in range(5):
        #short hill climb
        x0_best = hill_climber(C0, rho0, 0.01, 10, data, design_mat, n, m,
                               True)
        #minimize sum of squares with nelder mead
        res = minimize(ssq, x0_best, args = (rho, data, design_mat, n, m, True), 
                       method = 'nelder-mead', bounds = bounds_C, 
                       options = {'fatol':tol, 'maxiter':10000})
        #now fine tune with BFGS
        res = minimize(ssq, res.x, args = (rho, data, design_mat, n, m, True), 
                       method = 'BFGS', 
                       options = {'maxiter':10000})
        C0 = res.x
        SSQ = ssq(C0, rho0, data, design_mat, m, n, True)
        print('SSQ (C): ', SSQ)
        if res.fun < tol:
            i=9
        for i in range(5):
            #short hill climb
            rho0_best = hill_climber(rho0, C0, 1, 250, data, 
                                     design_mat, n, m, False)
            #minimize sum of squares with nelder mead
            res_rho = minimize(ssq, rho0_best, args = (C0, data, design_mat, n,
                                                       m, False), 
                               method = 'nelder-mead', bounds = bounds_rho, 
                               options = {'fatol':tol, 'maxiter':10000})
            #now fine tune with BFGS
            res_rho = minimize(ssq, res_rho.x, args = (C0, data, design_mat, n,
                                                       m, False), 
                               method = 'BFGS', 
                               options = {'maxiter':10000})
            rho0 = res_rho.x
            SSQ = ssq(rho0, C0,  data, design_mat, m, n, False)
            print('SSQ (rho): ', SSQ)
            if res_rho.fun < tol:
                break
    #before minimization
    plt.scatter(A2C(A, m, n).flatten(), C_cand.flatten(), c = 'grey')
    #after minimization
    plt.scatter(A2C(A, m, n).flatten(), res.x[:m*n],  c = 'black')
    plt.scatter(rho, rho0, c = 'black')
    plt.plot([-1, 3], [-1, 3])
    plt.show()
    return 0

## CODE ##

if (__name__ == '__main__'):
    status = main(sys.argv)
    sys.exit(status)
     

