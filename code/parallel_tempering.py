def sample_perturbation(vec, lb, ub):
    '''
    Sample perturbations respecting bounds

    Parameters:
        vec (1xn): vector to perturb
        lb (1xn): vector of lower bounds of variables in vector v
        ub (1xn): vector of upper bounds of variables in vector v
    '''
    n = len(vec)
    p_vec = np.zeros(n)
    for i in range(n): 
        p_vec[i] = np.random.uniform(lb[i]/vec[i]-1, ub[i]/vec[i]-1)
    return p_vec

def acceptance_probability(e, e_tmp, T):
    p = np.exp(-1/T*(e_tmp-e))
    return min(1, p)

def parallel_tempering(x0, lb, ub, T_vec):
    '''
    Perform parallel tempering on a vector ro find minimum

    Parameters:
        x0 (1xn): initial guess
        lb (1xn): lower bound for each variable
        ub (1xn): upper bound for each variable
        T_vec (1xm): vector of temperatures
        n_steps (int): number of steps of the chains
    '''
    #number of temperatures
    n_T = len(T_vec)
    #number of parameters
    n_p = len(x0)
    #preallocate matrix of solutions
    x_mat = np.tile(x0, n_T).reshape((n_T, n_p, 1))
    converged = 0
    import ipdb; ipdb.set_trace(context = 20)
    while not converged:
        #add a column of zeros to matrix of solutions
        x_mat = np.append(x_mat, np.zeros((n_p, 1)), axis = 2)
        #loop over temperatures
        for i in range(n_T): 
            #select appropriate parameter vector
            x = x_mat[i, :, -2]
            #compute initial ssq
            SSQ = ssq(x, observations, design_matrix, n, m, min_var, 
                      parameters)
            #sample perturbation for these parameters
            xi = sample_perturbation(x, lb, ub)
            #update next column with perturbation
            x_mat[i,:, -1] = x*(1+xi)
            #loop over each parameter
            for j in range(n_p):
                #modify parameter j
                x_tmp = x 
                x_tmp[j] = x_tmp[j]*(1+xi[j])
                #compute SSQ of perturbation
                SSQ_tmp = ssq(x_tmp, observations, design_matrix, n, m, min_var,
                              parameters)
                #calculate probability of acceptance
                p = acceptance_probability(SSQ, SSQ_tmp, T_vec[i])
                #throw coins with acceptance probability
                reject = np.random.binomial(1, 1-p)
                if reject:
                    x_mat[i, j, -1] = x_mat[i, j, -2]
        #every so often, shuffle the temperatures and accept or reject it
        #has it converged??
    #select the best solution based on the minimum SSQ
    
    return None

temps = np.array([2, 1, 0.5])

parallel_tempering()
