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
import scipy as sc
import pandas as pd
import matplotlib.pylab as plt
from scipy.optimize import minimize, Bounds, LinearConstraint
import progressbar
from taylor_community import perm_matrix_comb, all_comms
import itertools
sys.path.append('/home/pablo/Desktop/usefulprograms/code')
from essential_tools import *

## CONSTANTS ##


## FUNCTIONS ##

def C2A(C, D, l):
    '''
    Cast matrix C into matrix A
    '''
    m, n = C.shape
    A = np.block([[-np.identity(m), -C+l*D@C], 
                     [(1-l)*C.T, np.zeros((n, n))]])
    return A

def A2C(A, m, n):
    '''
    Extract matrix C given matrix A
    '''
    return A[0:m, m:m+n]

def simulate_data(n, m, design_matrix, l):
    '''
    Simulate feasible experimental observation by assemblying subcommunities 
    with a GLV

    Parameters:
        n: Number of spcies
        m: Number of resources
        design_matrix: matrix of observations
        l: leakage level
    '''
    feasible = False
    #sample growth rates and interactions
    while not feasible:
        #draw random parameters
        d = np.repeat(np.random.uniform(0, 100), n)
        r = 1+max(d)+np.random.uniform(0, 1, m)
        C = np.random.uniform(0, 1, size=(m,n))
        #Sample permutation matrix
        P_mat = perm_matrix_comb(m, 4)
        D = 1/2*(P_mat+P_mat.T)
        rho = np.concatenate((r, -d))
        #Compute B 
        B = np.eye(m) - l*D
        A = C2A(C, D, l)
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
            subcomm.assembly(t_dynamics=True)
            n_ext = len(subcomm.presence) - sum(subcomm.presence)
            print('Number of extinctions: ', n_ext)
            if any(subcomm.n == 0):
                break
            else: 
                #update row in matrix
                data[i,present] = subcomm.n
                if i==k-1: feasible = True
    return(data, A, rho, D)

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
        try:
            z_pred[i,present] = (np.linalg.inv(A_i)@rho[present, np.newaxis]).T
            return z_pred
        except:
            #If the matrix is singular
            return False


def zero_line(M):
    '''
    Find column or row of zeros in matrix M
    '''
    c_zeros = (~(M.any(axis = 0))).any()
    r_zeros = (~(M.any(axis = 1))).any()
    if c_zeros or r_zeros:
        return True
    else:
        return False

def ssq(x, observations, design_matrix, n, m, min_var, parameters):
    '''
    Minimization goal function

    Parameters:
        x: vector of values to minimize (either C, rho, or D)
        par1: one of the three, not x
        par2: the other one of the three, not x, not par1
        observations: observed abundances
        design_matrix (kxn array): Presence absence matrix for each species in
                                   each of the k experiemts.
        n: Number of species
        m: Number of resources
        min_var: name of the variable to minimize 
        parameters: dictionary of all parameters
    '''
    #Assign minimization variable and parameters
    if min_var == 'C':
        C_vec = x
        C = C_vec.reshape(m, n)
        rho = parameters['rho']
        B = parameters['B']
        l = parameters['l']
    elif min_var == 'rho':
        rho = x
        C = parameters['C']
        B = parameters['B']
        l = parameters['l']
    else:
        B_vec = x
        B = B_vec.reshape(m, n)
        C = parameters['C']
        rho = parameters['rho']
        l = parameters['l']
    #build matrix of interactions
    A = C2A(C, B, l)
    if zero_line(A):
        import ipdb; ipdb.set_trace(context = 20)
    #predict abundances
    z_pred = predict(A, rho, design_matrix)
    #if any abundance is negative, multiply its value as penalization
    ind_neg = np.where(z_pred<0)
    z_pred[ind_neg] += 10*z_pred[ind_neg] 
    #calculate sum of squares
    ssq = np.sum((z_pred - observations)**2)
    return ssq

def bound(x, lb, ub):
    if x < lb: 
        return lb
    elif x > ub:
        return ub
    else:
        return x

def swap_acceptance(e1, e2, T1, T2):
    p = np.exp((e1-e2)/np.mean(np.array([e1, e2]))*(1/T1 - 1/T2))
    return min(1, p)

def acceptance_probability(e, e_tmp, T):
    p = np.exp(-1/T*(e_tmp-e))
    return min(1, p)

def temperature(r, T0):
    '''
    return temperature when a fraction r of the total steps has passed
    '''
    return T0*(1-r)

def parallel_tempering(x0, lb, ub, T0_vec, n_steps, observations, 
                       design_matrix, n, m, min_var, parameters):
    '''
    Perform parallel tempering on a vector to find minimum

    Parameters:
        x0 (1xn): initial guess
        lb (1xn): lower bound for each variable
        ub (1xn): upper bound for each variable
        T0_vec (1xm): vector initial temperatures
        n_steps (int): number of cooling steps
    '''
    #number of chains
    n_c = len(T0_vec)
    #number of parameters
    n_p = len(x0)
    #preallocate matrix of solutions
    x_mat = np.tile(x0, n_c).reshape((n_c, n_p, 1))
    #preallocate ssq_measure for each chain, and step
    ssq_mat = np.zeros((n_c, n_steps))
    stuck = False
    #loop over temperatures to cool down each chain 
    for k in progressbar.progressbar(range(n_steps)): 
        #add a column of zeros to matrix of solutions
        x_mat = np.append(x_mat, 
                          np.tile(np.zeros((1, n_p)), 
                                  n_c).reshape(n_c, n_p, 1), axis = 2)
        #get vector of new temperatures
        T_vec = temperature(k/n_steps, T0_vec)
        #loop over chains
        for i in range(n_c): 
            #select appropriate parameter vector
            x = np.copy(x_mat[i, :, -2])
            #compute initial ssq
            SSQ = ssq(x, observations, design_matrix, n, m, min_var, 
                      parameters)
            #sample perturbation for these parameters
            xi = sc.stats.truncnorm.rvs(-1, 1, size=n_p)
            x_pert = x*(1+xi)
            #make sure parameters are within bounds
            x_pert = np.array([bound(x_pert[i], lb[i], ub[i]) \
                               for i in range(n_p)])
            #update next column with perturbation
            x_mat[i,:, -1] = x_pert
            #loop over parameters
            for j in range(n_p):
                #modify parameter j
                x_tmp = np.copy(x)
                x_tmp[j] = x_tmp[j]*(1+xi[j])
                #compute SSQ of perturbation
                SSQ_tmp = ssq(x_tmp, observations, design_matrix, n, m, 
                              min_var, parameters)
                #calculate probability of acceptance
                p = acceptance_probability(SSQ, SSQ_tmp, T_vec[i])
                #throw coins with acceptance probability
                reject = np.random.binomial(1, 1-p)
                if reject:
                    #switch back to unperturbed parameter
                    x_mat[i, j, -1] = x_mat[i, j, -2]
                else: 
                    #update SSQ
                    SSQ = ssq(x_tmp, observations, design_matrix, n, m, 
                              min_var, parameters)
            #update matrix element after perturbing all parameters
            ssq_mat[i, k] = ssq(x_mat[i, :, -1], observations,
                                design_matrix, n, m, min_var, parameters)
            if k > 10:
                #check if any chain is stuck
                stuck = is_stuck(ssq_mat[:, k-10:k])
        #switch two of the temperatures
        if stuck:
            rng = np.random.default_rng()
            swap_ind = rng.integers(n_c, size=2)
            p_swap = swap_acceptance(ssq_mat[swap_ind[0], k], 
                                     ssq_mat[swap_ind[1], k], 
                                     T_vec[swap_ind[0]],
                                     T_vec[swap_ind[1]])
            accept = np.random.binomial(1, p_swap)
            if accept:
                print('swapped')
                T0_vec[swap_ind[0]], T0_vec[swap_ind[1]] = T0_vec[swap_ind[1]],\
                                                           T0_vec[swap_ind[0]]


    #select the best solution based on the minimum SSQ
    return None

def hill_climber(x, magnitude, n_steps, observations, design_matrix, 
                 n, m, min_var, parameters):
    '''
    Plain-vanilla hill climber algorithm that modifies parameters randomly
    '''
    #initialize SSQ vector
    SSQ_vec = np.zeros(10)
    #compute initial ssq
    SSQ = ssq(x, observations, design_matrix, n, m, min_var, parameters)
    #add to vector
    SSQ_vec = add_element(SSQ, SSQ_vec)
    max_trials = 999
    #loop for decreasing values of perturbation
    for i in range(50):
        #perturbation trials keep going until SSQ can no longer be decreased
        trials = 0
        while not is_stuck(SSQ_vec) and trials < max_trials:
            trials += 1
            if trials > max_trials:
                print('max trials reached')
            if is_stuck(SSQ_vec):
                print('plateau reached')
            #sample perturbation
            p = np.random.normal(scale=magnitude, size=len(x))
            x_tmp = x*(1+p)
            SSQ_tmp = ssq(x_tmp, observations, design_matrix, n, m, min_var,
                          parameters)
            #keep perturbation if it decreases the error
            if SSQ_tmp < SSQ:
                x = x_tmp
                SSQ = SSQ_tmp
                SSQ_vec = add_element(SSQ, SSQ_vec)
                print('SSQ: ', SSQ)
        #decrease by 5% the magnitude of the perturbation
        magnitude *= 0.95
    return x

def get_ind_sub_comm(n, k):
    '''
    Get design matrix with desired subcommunities
    '''
    #Form a string with the number of species
    iterable = ''.join([str(i) for i in range(n)])
    #Form all the combinations of k species from the pool
    a = itertools.combinations(iterable, k)
    #transform into list of lists
    y = [[int(j) for j in i] for i in a]
    #initialize design matrix
    design = np.zeros((n, n))
    #build design matrix
    for i in range(n):
        design[i,y[i]] = 1
    return design

def get_constraint_mat(m):
    '''
    Construct matrix for doubly stochastic constraints
    
    Parameters:
        m (int): Number of resources
    '''
    #First part of the matrix
    M0 = sc.linalg.circulant(m*[1]+(m-1)*m*[0])
    #Second part of the matrix
    M1 = sc.linalg.circulant(m*([1]+(m-1)*[0]))
    #merge
    M = np.vstack((M0.T[0::m], M1.T[0:m]))
    return M

def add_element(element, v):
    '''
    Add element to a vector while keeping its length. If there are trailng
    zeroes, substitute those. Otherwise cut the first element, and add to the
    tail
    '''
    #get position of first zero
    try:
        first_0 = np.where(v==0)[0][0]
        v[first_0] = element
    except:
        #add at the end
        v = np.append(v, element)
        #delete first one
        v = np.delete(v, 0)
    return v


def is_stuck(SSQ_mat, tol=1e-2):
    '''
    Given the each of the last n elements of the SSQ vectors in SSQ matrix, 
    determine if any of them are not changing anymore.
    '''
    n_row, n_col = SSQ_mat.shape
    stuck_vec = np.zeros(n_row, dtype = bool)
    for i in range(n_row): 
        SSQ_i = SSQ_mat[i,:]
        #fit a line to vector of errors of row i
        slope, intercept = np.polyfit(np.arange(len(SSQ_i)), SSQ_i, 1)
        #declare stuck if slope is small enough
        if abs(slope) < tol:
            stuck_vec[i] = True
        else: 
            stuck_vec[i] = False
    if any(stuck_vec):
        return True
    else:
        return False


def main(argv):
    '''Main function'''
    #Set parameters
    n, m = (3,3)
    l = 0.1
    y = get_ind_sub_comm(n, 2)
    #create experimental design matrix
    spp_mat_monos = np.identity(n)
    spp_mat_doubles = get_ind_sub_comm(n, 2)
    spp_mat = np.vstack((spp_mat_monos, spp_mat_doubles))
    res_mat = np.ones((2*m, m))
    design_mat = np.hstack((res_mat, spp_mat))
    #generate data
    data, A, rho, D = simulate_data(n, m, design_mat, l)
    B = np.identity(m)-l*D
    #set tolerance
    tol = 1e-10
    #propose a C
    C_cand = np.random.uniform(0, 1, size=(m, n))
    #propose a rho
    d = np.repeat(np.random.uniform(0, 1), n)
    #Propose a B
    P_mat = perm_matrix_comb(m, 4)
    D = 1/2*(P_mat+P_mat.T)
    B0 = np.eye(m) - l*D
    r = 1+max(d)+np.random.uniform(0, 1, m)
    rho_cand = np.concatenate((r, -d))
    #set bounds
    bounds_C = Bounds(1e-9, 0.9999)
    bounds_rho = Bounds(m*[0]+n*[-np.inf], m*[np.inf]+n*[0])
    bounds_B = Bounds(m**2*[-1], m**2*[1])
    pars = {'C':C_cand, 'rho':rho, 'B':B, 'l':l}
    #parameters for parallel tempering
    temps = np.linspace(1000, 10, num = 50)
    x = parallel_tempering(C_cand.flatten(), m*n*[0], m*n*[1], temps, 1000,  
                           data, design_mat, n, m, 'C', pars)
    #hill climb first two parameters.
    C0 = hill_climber(C_cand.flatten(), 1, 250, data, design_mat, n, m, 'C', 
                      pars)
    #update parameter C
    pars['C'] = C0.reshape(m, n)
    rho0 = hill_climber(rho_cand.flatten(), 1, 250, data, design_mat, n, m, 
                        'rho', pars)
    #update parameter rho
    pars['rho'] = rho0
    #initialize SSQ vector
    SSQ_vec = np.zeros(10, dtype = int)
    #Estimate metabolic preferences matrix
    for i in range(5):
        #short hill climb
        x0_best = hill_climber(C0, 0.01, 10, data, design_mat, n, m, 'C', pars)
        #minimize sum of squares with nelder mead
        res = minimize(ssq, x0_best, 
                       args = (data, design_mat, n, m, 'C', pars), 
                       method = 'nelder-mead', bounds = bounds_C, 
                       options = {'fatol':tol, 'maxiter':10000})
        #now fine tune with BFGS
        #res = minimize(ssq, res.x, args = (data, design_mat, n, m, 'C', pars), 
        #               method = 'BFGS', 
        #               options = {'maxiter':10000})
        C0 = res.x
        pars['C'] = res.x.reshape(m, n)
        SSQ = ssq(res.x, data, design_mat, m, n, 'C', pars)
        print('SSQ (C): ', SSQ)
        if res.fun < tol:
            i=4
        #Estimate growth rates
        for j in range(5):
            #short hill climb
            rho0_best = hill_climber(rho0, 0.1, 250, data, design_mat, n, m, 
                                     'rho', pars)
            #minimize sum of squares with nelder mead
            res_rho = minimize(ssq, rho0_best, args = (data, design_mat, n, m, 
                                                       'rho', pars), 
                               method = 'nelder-mead', bounds = bounds_rho, 
                               options = {'fatol':tol, 'maxiter':10000})
            #now fine tune with BFGS
            res_rho = minimize(ssq, res_rho.x, args = (data, design_mat, n, m, 
                                                       'rho', pars), 
                               method = 'BFGS', 
                               options = {'maxiter':10000})
            rho0 = res_rho.x
            pars['rho'] = res_rho.x
            SSQ = ssq(res_rho.x, data, design_mat, m, n, 'rho', pars)
            print('SSQ (rho): ', SSQ)
            if res_rho.fun < tol:
                break
            #Estimate metabolic cross-feeding matrix
            for k in range(5):
                #matrix of constraints
                mat_constraint = get_constraint_mat(m)
                low = 1
                up = 1
                #set up constraint
                linear_constraint = LinearConstraint(mat_constraint, low, up)
                B0_vec = B0.flatten()
                #minimize
                res = minimize(ssq, B0_vec, args = (data, design_mat, 
                                                          n, m, 'B', pars),
                               method = 'trust-constr', 
                               constraints = [linear_constraint],
                               options={'verbose':0},
                               bounds = bounds_B)
                B0_vec = res.x
                pars['B'] = B0_vec.reshape(m, m)
                SSQ = ssq(res.x, data, design_mat, m, n, 'B', pars)
                print('SSQ (B): ', SSQ)
    #before minimization
    plt.scatter(A2C(A, m, n).flatten(), C_cand.flatten(), c = 'grey')
    #after minimization
    plt.scatter(A2C(A, m, n).flatten(), res.x[:m*n],  c = 'black')
    plt.scatter(rho, rho0, c = 'black')
    plt.plot([-1, 3], [-1, 3])
    plt.show()
    import ipdb; ipdb.set_trace(context = 20)
    return 0

## CODE ##

if (__name__ == '__main__'):
    try:
        status = main(sys.argv)
    except:
        import pdb, traceback
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
        sys.exit(status)
     

