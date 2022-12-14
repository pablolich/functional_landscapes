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

def A2C(A, m, n, l):
    '''
    Extract matrix C given matrix A
    '''
    lCT = A[m:2*m, 0:n]
    CT = lCT/(1-l)
    C = CT.T
    return C

def generate_doubly_stochastic(N, M0, epsilon=1e-6):
    '''
    Generate a doubly stochastic matrix of dimension N, starting form matrix 
    M0
    '''
    converged = False
    it_max = 20
    it = 0
    while not converged:
        #make the matrix row stochastic
        row_sum = np.sum(M0, axis = 1)
        M_row = M0/row_sum[:, None]
        #make the matrix column stochastic
        col_sum = np.sum(M_row, axis = 0)
        M_col = M_row/col_sum
        #check for convergence
        M0 = M_col
        it += 1
        if it > it_max:
            #using miminization to obtain doubly stochastic matrix
            mat_constr = get_constraint_mat(N)
            b = np.ones(2*N)
            res = minimize(constraint_eq, M0.flatten(), args=(mat_constr, b), 
                            method = 'Nelder-Mead', options={'fatol':epsilon})
            M0 = res.x.reshape(N, N)
            row_sum = np.sum(M0, axis = 1)
            col_sum = np.sum(M0, axis = 0)
        if all(row_sum - 1 < epsilon) and all(col_sum - 1 < epsilon):
            converged = True
    return M0

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
        d = np.random.uniform(0, 100, n)
        r = 1+max(d)+np.random.uniform(0, 1, m)
        C = np.random.uniform(0, 1, size=(m,n))
        #Sample permutation matrix
        P_mat = perm_matrix_comb(m, 4)
        D = 1/2*(P_mat+P_mat.T)
        rho = np.concatenate((r, -d))
        #Compute B 
        B = np.eye(m) - l*D
        A = C2A(C, D, l)
        A = np.array(pd.read_csv('../data/A.csv'))
        rho = np.array(pd.read_csv('../data/rho.csv')).flatten()
        D = np.array(pd.read_csv('../data/D.csv'))
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
            x_star = -np.linalg.inv(subcomm.A)@subcomm.rho
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
            z_pred[i,present] = -(np.linalg.inv(A_i)@rho[present, np.newaxis]).T
        except:
            #If the matrix is singular
            return False
    return z_pred


def vec2D(vec, m):
    '''
    Trsansform upper triangular vector of D into symmetric D
    '''
    i_upper = np.triu_indices(m)
    D = np.zeros((m, m))
    D[i_upper] = vec
    i_lower = np.tril_indices(m, -1)
    D[i_lower]=D.T[i_lower]
    return D

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
        D = parameters['D']
        l = parameters['l']
    elif min_var == 'rho':
        rho = x
        C = parameters['C']
        D = parameters['D']
        l = parameters['l']
    elif min_var == 'D':
        n_D = int(1/2*m*(m+1))
        D = vec2D(x, m)
        C = parameters['C']
        rho = parameters['rho']
        l = parameters['l']
    elif min_var == 'Candrho':
        #minimize at the same time
        C_vec = x[0:m*n]
        C = C_vec.reshape(m, n)
        rho = x[m*n:m*n+m+n]
        D = parameters['D']
        l = parameters['l']
    else:
        #minimize at the same time
        C_vec = x[0:m*n]
        C = C_vec.reshape(m, n)
        rho = x[m*n:m*n+m+n]
        n_D = int(1/2*m*(m+1))
        D = vec2D(x[m*n+m+n:m*n+m+n+n_D], m)
        l = parameters['l']
    #build matrix of interactions
    A = C2A(C, D, l)
    #predict abundances
    z_pred = predict(A, rho, design_matrix)
    #if any abundance is negative, multiply its value as penalization
    #ind_neg = np.where(z_pred<0)
    #z_pred[ind_neg] += 1e4*z_pred[ind_neg] 
    #calculate sum of squares
    ssq = np.sum((z_pred - observations)**2)
    return ssq

def bound(x0, x, lb, ub):
    if x < lb or x > ub: 
        return 2*x0-x
    else:
        return x

def swap_accept(e1, e2, T1, T2, ind):
    p = np.exp((e1-e2)*(1/T1 - 1/T2))
    return min(1, p)

def acceptance_probability(e, e_tmp, T):
    p = np.exp(-1/T*(e_tmp-e))
    return min(1, p)

def piggyback_swap(chain, T0_vec, T_vec, e_vec, ind):
    '''
    Recursively try to swap temperatures from below with its neighbours
    Parameters:
        chain: index of the chain we are attempting to swap with its neighbour
        T_vec: vector of temperatures
        e_vec: vector of errors
    '''
    while chain > 0:
        k = np.mean(e_vec)
        k=1
        #if ind==350:
        #    import ipdb; ipdb.set_trace(context = 20)
        #get swapping probability
        p_swap = swap_accept(e_vec[chain], e_vec[chain-1],
                             T_vec[chain], T_vec[chain-1], k, ind)
        #throw a coin
        should_swap = np.random.binomial(1, p_swap)
        if should_swap:
            #perform swap
            T0_vec[chain], T0_vec[chain-1] = T0_vec[chain-1], T0_vec[chain]
            T_vec[chain], T_vec[chain-1] = T_vec[chain-1], T_vec[chain]
            chain -= 1
            return piggyback_swap(chain, T0_vec, T_vec, e_vec, ind)
        else:
            chain -= 1
            return piggyback_swap(chain, T0_vec, T_vec, e_vec, ind)
    return T0_vec

def temperature(x, x_f, T0, p):
    '''
    return temperature when a fraction r of the total steps has passed
    '''
    return T0*(1-(x/x_f)**p)**(1/p)

def random_swap(T0_vec, T_vec, e_vec, ind):
    '''
    Swap two random chains, if its favorable (according to probability)
    '''
    n_T = len(T0_vec)
    k = 1
    #sample two random indices to swap
    inds = np.random.choice(np.arange(n_T), 2, replace = False)
    #calculate probability of swapping
    p_swap = swap_accept(e_vec[inds[0]], e_vec[inds[1]],
                         T_vec[inds[0]], T_vec[inds[1]], k, ind)
    #throw the swapping coin
    should_swap = np.random.binomial(1, p_swap)
    if should_swap:
        #perform swap
        T0_vec[inds[0]], T0_vec[inds[1]] = T0_vec[inds[1]], T0_vec[inds[0]]
        T_vec[inds[0]], T_vec[inds[1]] = T_vec[inds[1]], T_vec[inds[0]]
    return T0_vec

def n_pert(T, n_param, T_max):
    n_per = int(np.ceil(n_param*T/T_max))
    if n_per > n_param:
        n_per = n_param
    return n_per

def get_perturbation_matrix(m):
    '''
    get a doubly stochastic symmetric matrix 
    '''
    P = perm_matrix_comb(m, 2)
    return 1/2*(P + P.T)

def consecutive_swap(T0_vec, ind):
    '''
    Swap chain 'ind' with chain 'ind-1'
    '''
    T0_vec[ind], T0_vec[ind-1] = T0_vec[ind-1], T0_vec[ind]
    return T0_vec

def parallel_tempering(x0, lb, ub, T0_vec, n_steps, p, observations, 
                       design_matrix, n, m, min_var, parameters):
    '''
    Perform parallel tempering on a vector to find minimum
    Parameters:
        x0 (1xn): initial guess
        lb (1xn): lower bound for each variable
        ub (1xn): upper bound for each variable
        T0_vec (1xm): vector initial temperatures
        n_steps (int): number of cooling steps
        p (float): convexity of the cooling schedule
    '''
    #number of chains
    n_c = len(T0_vec)
    #number of parameters of each group of objects to minimize 
    n_C_rho = m*n+m+n
    n_D = int(m*(m+1)//2)
    #number of parameters
    n_p = n_C_rho + n_D 
    #preallocate matrix of solutions
    x_mat = np.tile(x0, n_c).reshape((n_c, n_p, 1))
    #preallocate ssq_measure for each chain, and step
    ssq_mat = np.zeros((n_c, n_steps))
    #preallocate big-ass dataframe for plotting in r
    n_rows = n_steps*n_c
    col_names = ['t', 'chain', 'T', 'ssq']
    df = pd.DataFrame(data = np.zeros((n_rows,len(col_names))), 
                      columns = col_names)
    ind_df = 0
    B_cum = np.zeros((n_c, m, m))
    n_accepts = np.zeros((n_c, n_steps))
    #loop over temperatures to cool down each chain 
    for k in progressbar.progressbar(range(n_steps)):
        #add a column of zeros to matrix of solutions
        x_mat = np.append(x_mat, 
                          np.tile(np.zeros((1, n_p)), 
                                  n_c).reshape(n_c, n_p, 1), axis = 2)
        #get vector of new temperatures
        T_vec = temperature(k, n_steps, T0_vec, p)
        #loop over chains
        for i in range(n_c): 
            #select appropriate parameter vector
            x = np.copy(x_mat[i, :, -2])
            #compute initial ssq
            SSQ = ssq(x, observations, design_matrix, n, m, min_var, 
                      parameters)
            #get number of parameters to perturb
            n_per = n_pert(T_vec[i], n_C_rho, max(T0_vec))
            #sample perturbation for these parameters
            np.random.seed(i+k)
            xi = np.random.uniform(0.9, 1.1, size=n_per)
            #determine which parameters to perturb randomly
            ind_per = np.sort(np.random.default_rng(i+k).choice(np.array(n_C_rho), 
                                                                n_per, 
                                                                replace=False))
            x_pert = np.copy(x)
            x_pert[ind_per] = x_pert[ind_per]*xi
            #make sure parameters are within bounds
            x_pert = np.array([bound(x[i], x_pert[i], lb[i], ub[i]) \
                               for i in range(n_p)])
            #now sample perturbation for D
            pert_D = get_perturbation_matrix(m)
            x_pert[n_C_rho:n_C_rho+n_D] = 0.9*x_pert[n_C_rho:n_C_rho+n_D]+ \
                                          0.1*pert_D[np.triu_indices(m)].flatten()
            #compute SSQ of perturbation
            SSQ_pert = ssq(x_pert, observations, design_matrix, n, m, 
                           min_var, parameters)
            #calculate probability of acceptance
            p_accept = acceptance_probability(SSQ, SSQ_pert, T_vec[i])
            #determine if we accept the perturbation
            np.random.seed(i+k)
            reject = np.random.binomial(1, 1-p_accept)
            if reject:
                #update with unperturbed parameters
                x_mat[i, :, -1] = np.copy(x_mat[i, :, -2])
                n_accepts[i, k] = n_accepts[i, k-1]
            else: 
                #update with perturbed parameters
                #update next column with perturbation
                x_mat[i,:, -1] = x_pert
                B_cum[i] += pert_D
                n_accepts[i, k] = n_accepts[i, k-1] + 1
                #update SSQ
                SSQ = SSQ_pert
            #update matrix element of SSQ
            ssq_mat[i, k] = SSQ
            vec_store = np.array([k, i, T_vec[i], SSQ])
            df.iloc[ind_df,:] = vec_store
            ind_df += 1
        #swap every 50 iterations
        if k % 50 == 0 and k > 0:
            #get vector of errors for each chain
            err_vec = ssq_mat[:, k]
            #sample a chain
            ind_chain = np.random.choice(np.arange(n_c-1)+1)
            #calculate probability of acceptance
            p_chain_swap = swap_accept(err_vec[ind_chain], 
                                       err_vec[ind_chain-1], 
                                       T_vec[ind_chain], T_vec[ind_chain-1], k)
            #decide if I swap
            swap = np.random.binomial(1, p_chain_swap)
            if swap:
                #perform swap
                T0_vec = consecutive_swap(T0_vec, ind_chain)
            

    return x_mat, ssq_mat, df, n_accepts

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

def get_constraint_symmetric_bistochastic(m):
    #elements in the upper triangle
    n = m*(m+1)//2
    #initialize matrix
    M = np.zeros((m, n))
    for i in range(m):
        for j in range(m):
            ind_i = np.copy(i)
            ind_j = np.copy(j)
            if ind_i > ind_j:
                ind_i, ind_j = ind_j, ind_i
            k = m*ind_i + ind_j - int(ind_i*(ind_i+1)/2)
            M[i, k] = 1
    return M

def bottom_up_inference(C, D, leak, inds, attractors):
    '''
    Infer attractors of high dimensional communities given those of lower 
    dimmensional systems

    Parameters:
        C (mxn): Matrix of consumer preferences
        D (mxm): Matrix of metabolic crossfeeding
        inds (lxk): indices of present species in each of the l subcommunities
        attractors (mxl): attractors of each of the l subcommunities
    '''
    l = len(inds)
    B = np.identity(len(D))- leak*D
    #initialize vector of things
    vec=np.array([])
    for i in range(l):
        add = np.array([C[:, inds[i]].T[0]@B@attractors[:,i]])
        vec = np.concatenate((vec, add))
    f = C@np.linalg.inv(C.T@B@C)@vec
    return f

def top_down_inference(C, D, ind, attractor):
    '''
    Infere attractors from lower dimensional communities given the single
    attractor of a larger community

    Parameters:
        C (mxn): Matrix of consumer preferences
        D (mxm): Matrix of metabolic crossfeeding
        ind (1xk): indices of present species in the subcommunity
        attractor (1xm): attractors of each of the subcommunity
    '''
    #get submatrix C_k
    C_k = C[:,ind] 
    #compute projector matrix
    P = C_k@np.linalg.inv((C_k.T@C_k))@C_k.T
    f_k = P@attractor[:, None]
    return f_k

def get_attractor(C, x):
    '''
    Compute functional attractor given the preferences of a subcommunity and 
    the abundances at equilibrium
    '''
    return C@x[:, None]

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
    #data, A, rho, D = simulate_data(n, m, design_mat, l)
    data = np.array(pd.read_csv('../data/data.csv'))
    A = np.array(pd.read_csv('../data/A.csv'))
    rho = np.array(pd.read_csv('../data/rho.csv')).flatten()
    D = np.array(pd.read_csv('../data/D.csv'))
    C = A2C(A, m, n, l)
    #set random seed
    np.random.seed(1)
    #initial guesses
    C_cand = np.random.uniform(0, 1, size=(m, n))
    D_cand = generate_doubly_stochastic(m, np.random.uniform(0,1,size=(m,m)))
    #make symmetric
    D_cand = 1/2*(D_cand+D_cand.T)
    d = np.random.uniform(0, 100, n)
    r = 1+max(d)+np.random.uniform(0, 1, m)
    rho_cand = np.concatenate((r, -d))
    #number of D parameters in the upper triangular part
    n_D = int(1/2*m*(m+1))
    #set parameter bounds
    bounds_C = Bounds(1e-9, 0.9999)
    bounds_rho = Bounds(m*[0]+n*[-np.inf], m*[np.inf]+n*[0])
    boundsCrho = Bounds(m*n*[0]+m*[0]+n*[-np.inf], 
                        m*n*[1]+m*[np.inf]+n*[0])
    bounds_D = Bounds(n_D*[0], n_D*[1])
    #make dictionary
    pars = {'C':C_cand, 'rho':rho_cand, 'D':D_cand, 'l':l}
    #parameters for parallel tempering
    temps = np.array([1,10, 100, 1000])
    n_chains = len(temps)
    n_steps = 100000
    #bounds for varibles
    lowerbounds = m*n*[0] + m*[0] + n*[-np.inf] + n_D**2*[0]
    upperbounds = m*n*[1] + m*[np.inf] + n*[0] + n_D**2*[1]
    #get ind upper diagonal
    ind_u = np.triu_indices(m)
    #make initial guess vector
    x_vec = np.concatenate((C_cand.flatten(), rho_cand, 
                            D_cand[ind_u].flatten()))
    x_vec0 = np.copy(x_vec)
    #set up constraints
    mat_constraint = get_constraint_symmetric_bistochastic(m)
    low = 1
    up = 1
    linear_constraint = LinearConstraint(mat_constraint, low, up)
    #set tolerance
    tol = 15000
    #preallocate storing and ssq
    df_tot = pd.DataFrame(columns = ['t', 'chain', 'T', 'ssq'])
    best_ssq = np.inf
    while abs(best_ssq) > tol:
        #run parallel tempering
        x_mat, ssq_mat, df, acc_rate = parallel_tempering(x_vec, lowerbounds, 
                                                          upperbounds, temps, n_steps, 
                                                          .8, data, design_mat, n, m, 
                                                          'Candrho', pars)
        for i in range(n_chains):
            plt.plot(np.arange(n_steps), acc_rate[i,:]/n_steps)
        plt.show()
        #store in dataframe
        old_time = np.unique(np.array(df_tot['t']))
        new_time = np.unique(np.array(df['t']))
        if not df_tot.empty:
            new_time = 1+np.concatenate((np.array([new_time[0]]), new_time))
            max_chain = 1+max(df_tot['chain'])
            df['chain'] = df['chain'] + max_chain
        concat_time = cumulative_storing(old_time, new_time, time = True)[0]
        df_tot = pd.concat([df_tot, df]) 
        df_tot['t'] = np.repeat(concat_time, n_chains)
        #get min ssq
        ind_min = np.argmin(ssq_mat[:, -1])
        best_ssq = ssq_mat[ind_min, -1]
        print(best_ssq)
        #vector for next round
        x_vec = x_mat[ind_min,:, -1]
        #update parameters
        pars['C'] = x_vec[0:m*n].reshape(m, n)
        pars['rho'] = x_vec[m*n:m*n+m+n]
        D_best = vec2D(x_vec[m*n+m+n:m*n+m+n+n_D], m)
        pars['D'] = D_best
        #refine estimations of C and rho with nelder mead
        Crho_best = x_vec[0:m*n+m+n]
        res = minimize(ssq, Crho_best, 
                       args = (data, design_mat, n, m, 'Candrho', pars), 
                       method = 'nelder-mead', bounds = boundsCrho, 
                       options = {'fatol':1e-6, 'disp':True})
        if res.success:
            pars['C'] = res.x[0:m*n].reshape(m, n)
            pars['rho'] = res.x[m*n:m*n+m+n]
            x_vec[0:m*n+m+n] = res.x
            best_ssq = res.fun
            print(best_ssq)
        D_best = pars['D']
        #refine estimation of D with trust-constr algorithm
        res = minimize(ssq, D_best[ind_u].flatten(), 
                       args = (data, design_mat, n, m, 'D', pars),
                       method = 'trust-constr', 
                       constraints = [linear_constraint],
                       options={'verbose':3},
                       bounds = bounds_D)
        if res.success:
            x_vec[m*n+m+n:m*n+m+n+n_D] = res.x
            D_best = vec2D(res.x, m)
            pars['D'] = D_best
            best_ssq = res.fun
            print(best_ssq)
        temps = temps/2
        #decrease temperatures
        if max(temps) < 1e-3:
            temps = np.random.uniform(0,2,size=len(temps))
    df_tot.to_csv('../data/tot_results.csv', index=False)
    x0_best = x_vec
    #save my best guess
    np.savetxt('../data/best_guess.csv', x0_best, delimiter = ',')
    vec_original = np.concatenate((C.flatten(), rho, D[ind_u].flatten()))
    plt.scatter(vec_original[0:m*n],  x0_best[0:m*n], c='g')
    plt.scatter(vec_original[m*n:m*n+m+n], x0_best[m*n:m*n+m+n], c='r')
    plt.scatter(vec_original[m*n+m+n:], x0_best[m*n+m+n:], c='k')
    plt.scatter(vec_original, x_vec0)
    plt.plot(np.array([min(np.concatenate((x_vec, vec_original))),
                       max(np.concatenate((x_vec, vec_original)))]),
             np.array([min(np.concatenate((x_vec, vec_original))),
                       max(np.concatenate((x_vec, vec_original)))]))
    plt.show()
    #Form community with given species
    glv_comm = Community(np.array([1,1,1]), GLV, A = C2A(C, D, l), rho = r)
    inds = np.identity(n, dtype = bool)
    #form attractors
    attractors = np.zeros((m, n))
    for i in range(n):
        attractors[:,i] = get_attractor(C[:, inds[i]], 
                                        data[i, m:][inds[i]]).flatten()
    #Get original attractors
    l = 0.1
    v_theo = bottom_up_inference(C, D, l, inds, attractors)
    #load best inference
    x_best = np.loadtxt('../data/best_guess.csv')
    C = x_best[:m*n].reshape(m, n)
    D = vec2D(x_best[m*n+m+n:], m)
    v_infer = bottom_up_inference(C, D, l, inds, attractors)
    plt.plot(np.array([min(np.concatenate((v_infer, v_theo))),
                       max(np.concatenate((v_infer, v_theo)))]),
             np.array([min(np.concatenate((v_infer, v_theo))),
                       max(np.concatenate((v_infer, v_theo)))]))
    plt.scatter(v_infer, v_theo)
    plt.show()
    

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
     
