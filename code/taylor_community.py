__appname__ = '[taylor_community.py]'
__author__ = 'Pablo Lechon (plechon@ucm.es)'
__version__ = '0.0.1'

## IMPORTS ##

import sys
import numpy as np
import pandas as pd
from scipy.stats import beta, dirichlet
from itertools import combinations
import math
from scipy.spatial import distance
import matplotlib.pylab as plt
from scipy.optimize import minimize, Bounds, LinearConstraint
import statsmodels.api as sm
sys.path.append('/home/pablo/Desktop/usefulprograms/code')
from essential_tools import *


## CONSTANTS ##


## FUNCTIONS ##

def dec2bin(x):
    '''
    get binary number representation of a number
    '''
    return list(map(int, bin(x)[2:]))

def all_comms(n):
    '''
    Get all 2**n -1 (leave the bare ground out) possible combination of n 
    species in binary
    '''
    n_rows = 2**n - 1
    vec = np.zeros(shape=(n_rows, n), dtype = int)
    binary_string = np.zeros(shape=(n_rows, 1), dtype = '<U'+str(n))
    i = 0
    while i < n_rows:
        #transform to binary (start at 1 to leave out the bare ground)
        binary = dec2bin(int(i+1))
        #complete with zeros
        vec[i, :] = np.hstack((np.zeros(n - len(binary)), binary))
        i += 1
    return vec

def traverse_index(x, n):
    '''
    form all combinations of values of vector x that are, at most, n positions
    apart
    '''
    ind_vec = np.arange(len(x))
    #get all combinations of vector elements
    all_comb = np.meshgrid(ind_vec, ind_vec)
    #form matrix with indexed diagonals
    diagonal_ind = abs(all_comb[0] - all_comb[1])
    #get indices of nth first diagonals of matrix m, where 0 is the main 
    #diagonal, 1, is the first top and bottom off diagonals, etc...
    diag_index = np.where(diagonal_ind <= n)
    return diag_index

def generate_parameters(source):
    #traverse shapes of interest in the beta distribution in a clever way
    ind_loop = traverse_index(source, 1)
    n_shape = len(ind_loop[0])
    a_vec = np.zeros(n_shape)
    b_vec = np.zeros(n_shape)
    #get correct indices
    for i in range(n_shape):
        a_vec[i] = source[ind_loop[0][i]] 
        b_vec[i] = source[ind_loop[1][i]]
    return(a_vec, b_vec)

def randbin(n, m, count):
    '''
    Create a nxm binary random matrix conditioned to the constrained that there
    need be "count" ones in each row.
    '''
    #output to assign ones into
    result = np.zeros((n, m))
    #simulate sampling with replacement in one axis
    col_ind = np.argsort(np.random.random(size=(n, m)), axis=1)
    #turn it into a mask over col_ind using a clever broadcast
    try:
        mask = np.arange(m) < np.round(count).reshape(n, 1)
    except:
        mask = np.arange(m) < np.round(count)
    #apply the mask not only to col_ind, but also the corresponding row_ind
    col_ind = col_ind[mask]
    row_ind = np.broadcast_to(np.arange(n).reshape(-1, 1), (n, m))[mask]
    #Set the corresponding elements to 1
    result[row_ind, col_ind] = 1
    return result

def optimal_abundances(C_k, C, B, x):
    x = x.reshape(len(x), 1)
    return (np.linalg.inv(C_k @ B @ C_k.T) @ C_k @ B @ C.T @ x).T[0]

def comm_function(C, abundances):
    x = abundances.reshape(len(abundances), 1)
    return (C.T@x).T[0]

def vec_enlarge(vector, dimension, indices):
    '''
    Embed elements of a vector at positions of a higher dimmensional vector of
    zeros
    '''
    #output vector
    vec_out = np.zeros(dimension)
    vec_out[indices] = vector 
    return list(vec_out)

def generate_doubly_stochastic(N, M0, epsilon=1e-9):
    '''
    Generate a doubly stochastic matrix of dimension N, starting form matrix 
    M0
    '''
    converged = False
    #repeat the following until convergence
    D1 = np.eye(N)
    D2 = np.eye(N)
    while not converged:
        #make the matrix row stochastic
        Dv = 1/np.sum(M0, axis=1)*np.eye(N)
        M_row = Dv @ M0
        Dw = 1/np.sum(M_row, axis=0)*np.eye(N)
        #make the matrix column stochastic
        M_col = M_row @ Dw
        #Record matrices
        D1 *= Dv
        D2 *= Dw
        #get sum of columns
        col_sum = np.sum(M_col, axis = 0)
        row_sum = np.sum(M_col, axis = 1)
        #check for convergence
        converge_col = np.all((col_sum > 1-epsilon) & (col_sum < 1 + epsilon))
        converge_row = np.all((row_sum > 1-epsilon) & (row_sum < 1 + epsilon))
        converged = np.all((converge_col) & (converge_row))
        M0 = M_col
    #Find the constant relating them both
    w = np.diag(D2)/np.diag(D1)
    #Find unique D
    D = w**(1/2)*D1
    return D

def perm_matrix_comb(n, N):
    '''
    Convex combination of permutation matrices
    '''
    #Sample N permutations
    perm_mat = np.zeros(shape=(N, n), dtype = int)
    for samp in range(N):
        perm_mat[samp,:] = np.random.choice(np.arange(n), n, replace=False)
    #Initialize all permutation matrices
    A_all = np.zeros(shape=(N, n, n))
    for perm in range(N):
        for row in range(n):
            A_all[perm, row, perm_mat[perm][row]] = 1
    #Sample a stochastic vector
    vec = np.random.dirichlet(np.ones(N))
    #Perform combex combination of all matrices
    A = np.sum(A_all*vec[:, np.newaxis, np.newaxis], axis = 0)
    return A

def sum_D_xi(D, xi):
    m = len(D)
    result = 0
    for i in np.arange(m):
        for j in np.arange(m):
            if j > i:
                result += D[i, j]*(xi[i] - xi[j])**2
    return result

def obj_func(x, *args):
    l = args[0]
    C = args[1]
    D = args[2]
    xstar = args[3].reshape(len(args[3]), 1)
    C_k = args[4]
    xi = (C.T@xstar - C_k.T@x.reshape(len(x), 1)).T[0]
    S = sum_D_xi(D, xi)
    return (1-l)*np.dot(xi, xi) + l*S

def is_positive_definite(A):
    eig = np.linalg.eigvals(A)
    return np.all(eig>0)

def main(argv):
    '''
    perform taylor expansion of a community. pick a community function, and get
    the minimum number of species that will afford you the same function 
    within a certain tolerance. 
    Analyze how the error decreases as we add more species, and if having 
    differences in the resource occupancy spectrum yields qualitatively 
    different results
    '''
    #set parameters
    n, m = (int(sys.argv[1]), int(sys.argv[2]))
    l_str = sys.argv[3].split(',')
    l_vec = np.array([float(i) for i in l_str])
    col_names = ['sim', 'N', 'n', 'leakage',  
                 'distance_assembly', 'distance_optimal', 'distance_naive', 
                 'dist_SSE', 'dist_D', 'R2', 'R2_adj', 'spp_name', 
                 'ab_original', 'ab_optimal', 'ab_subcomm', 'ab_naive']
    col_names = col_names + ['r'+str(i+1)  for i in range(m)]
    df = pd.DataFrame(columns = col_names)
    it = 0
    n_sim = int(sys.argv[4])
    save_name = sys.argv[5]
    sim = 0
    #perform replicates
    while sim < n_sim:
        #Set growth and death rates for each simulation
        d = min(1-l_vec)*np.repeat(np.random.uniform(0, 1), n)
        r = 1+max(d)+np.random.uniform(0, 1, m)
        a = 1
        b = 1
        #sample number of preferences of each species from beta 
        #distribution with parameters a, b
        n_pref = beta.rvs(a, b, loc=1, scale=m-1, size=n)
        #build preference matrix
        C = randbin(n, m, n_pref)    
        #normalize it
        C = (C.T * 1/np.round(n_pref)).T
        #Sample C from a dirichlet distribution
        #C = np.random.dirichlet((np.ones(m)), n)
        #Get crossfeeding matrix
        M = np.random.random((m, m))
        #Make symmetric and positive definite
        A = M@M.T
        P = generate_doubly_stochastic(m, A)
        D = P@A@P
        #Another alternative
        #Sample permutation matrix
        P_mat = perm_matrix_comb(m, 4)
        D = 1/2*(P_mat+P_mat.T)
        posdef = is_positive_definite(D)
        #Save D
        pd.DataFrame(D).to_csv('../data/D.csv', index = False)
        #Vary the leakage level
        for l in l_vec:
            #Compute B 
            B = np.eye(m) - l*D
            #get parameters of equivalent lotka volterra
            I = np.identity(m)
            A = (1-l)*(C@B@C.T)
            rho = (1-l)*C@r-d
            #initialize community object
            glv_community = Community(np.ones(n), GLV, A=A, r=rho)
            #assembly community and forget its history
            glv_community.assembly()
            if not glv_community.converged:
                print('Assembly did not converge')
                continue
            #remove preferences from extinct species
            C_ext = C[glv_community.presence, :]
            #get new number of preferences vector after assembly
            n_pref_eff = [min(1/C_ext[i,:]) for i in range(len(C_ext))]
            #calculate mean and variance of the beta distribution sample
            mean = np.mean(n_pref_eff)
            var = np.var(n_pref_eff)
            #forget assembly history
            glv_community = glv_community.delete_history()
            #2. Measure function
            f = comm_function(C_ext, glv_community.n)
            #3. Among all the N choose n combinations of subcommunities of n 
            #   species, find that which minimizes the difference between the 
            #   whole community function and its subcommunity function
            #get all subcommunities of current one
            vec = all_comms(glv_community.richness)
            #get number of species in each subcommunity
            n_spp = np.sum(vec, axis=1)
            for j in range(glv_community.richness):
                #let assembly determine this abundances, and then select the 
                #best one. 
                n_spp_j = j + 1
                #select all subcommunities with j species
                ind = np.where(n_spp == n_spp_j)[0]
                sub_comms = vec[ind, :]
                n_sub = math.comb(glv_community.richness, n_spp_j)
                for sub in range(n_sub):
                    #get indices of species to be removed
                    try:
                        ind_rem = np.where(sub_comms[sub,:] == 0)[0]
                    except:
                        import ipdb; ipdb.set_trace(context = 20)
                    #form subcommunity
                    glv_sub_comm = glv_community.remove_spp(ind_rem)
                    #assembly the subcommunity
                    glv_sub_comm.assembly()
                    #only work with those communities with j species after 
                    #assembly
                    if glv_sub_comm.richness == n_spp_j: 
                        #get abundances
                        abundances = np.array(glv_sub_comm.n)
                        #remove preferences from extinct species
                        C_sub = C_ext[glv_sub_comm.presence, :]
                        #measure function
                        f_sub = comm_function(C_sub, abundances)
                        #get error between original funct and sub-comm funct
                        dist_assem = (distance.mahalanobis(f_sub, f, B))**(2)
                        #record abundances pre-assembly
                        ab_bare = glv_community.n[glv_sub_comm.presence]
                        #get function of bare abundances 
                        f_bare = comm_function(C_sub, ab_bare)
                        #calculate error with original one
                        dist_bare = (distance.mahalanobis(f_bare, f, B))**(2)
                        #Compute abundances using a GLS model
                        gls_model = sm.GLS(f, C_sub.T, sigma=np.linalg.inv(B))
                        gls_results = gls_model.fit()
                        #Get abundances
                        ab_gls = gls_results.params
                        #Get R**2 and adj R**2
                        r2 = gls_results.rsquared
                        adj_r2 = gls_results.rsquared_adj
                        #get sum of square of residuals from the fit
                        res = gls_results.resid
                        f_opt = gls_results.predict()
                        dist_opt = distance.mahalanobis(f, f_opt, B)**2
                        dist_SSE = np.dot(res, res)
                        dist_D = (res@D@res[:,np.newaxis])[0]
                        #complete vector of bare, assembly, and optimal 
                        #abundances of subcommunity with the removed 
                        #species for later storage
                        ab_sub = vec_enlarge(abundances,
                                             glv_community.richness,
                                             glv_sub_comm.presence)
                        ab_optim = vec_enlarge(ab_gls, 
                                               glv_community.richness,
                                               glv_sub_comm.presence) 
                        ab_bare = vec_enlarge(ab_bare,
                                              glv_community.richness,
                                              glv_sub_comm.presence)
                        sys.stdout.write("\033[K")
                        print('Running simulation', sim, 
                              'for leakage = ', l,' N = ', 
                              glv_community.richness, ', n = ', n_spp_j, 
                              'and distance = %.3f' % dist_assem, 
                              'checked: ', sub, '/',n_sub, end = '\n')
                        sys.stdout.write("\033[K")
                        vec_store = np.array([sim, glv_community.richness, 
                                              n_spp_j, l, dist_assem, dist_opt,
                                              dist_bare, dist_SSE, dist_D, r2, 
                                              adj_r2])
                        #replicate to store species-specific information in 
                        #long format
                        mat_store = np.tile(vec_store, 
                                            glv_community.richness).\
                                            reshape(glv_community.richness, 11)
                        #transform to dictionary
                        dict_store = {col_names[i]:list(mat_store[:,i]) for i 
                                      in range(mat_store.shape[1])}
                        #add species-specific columns
                        #add species name
                        dict_store['spp_name'] = ['spp'+str(i+1) \
                                                  for i in \
                                                  range(glv_community.richness)]
                        dict_store['ab_original'] = list(glv_community.n)
                        dict_store['ab_optimal'] = ab_optim
                        dict_store['ab_subcomm'] = ab_sub
                        dict_store['ab_naive'] = ab_bare
                        for i in range(m):
                            dict_store['r'+str(i+1)] = list(C_ext[:, i])
                        df_append = pd.DataFrame(dict_store)
                        df = pd.concat([df, df_append], axis=0)
                        #overwrite after each iteration
                        #update index of dataframe
                        it += 1
                    else: 
                        #skip iteration when there are extinctions
                        continue
        sim += 1
        df.to_csv('../data/'+save_name, index = False)
    return 0

## CODE ##

if (__name__ == '__main__'):
    status = main(sys.argv)
    sys.exit(status)
     

