###############################################################################
#IMPORTS
import copy
import os
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pylab as plt
###############################################################################

###############################################################################
#MODELS
def GLV(t, x, A, r, tol):
    '''
    n-species Generalized Lotka-Volterra

    Parameters:
        x (nx1): Species abundances
        A (nxn): Interaction matrix
        r (nx1): Species growth rates
        tol (float): solution precision
    '''
    return (x*(r + A@x)).T

###############################################################################

###############################################################################
#CLASSES
class Community:
    def __init__(self, n, model, A=None, C=None, D=None, r=None, rho=None, 
                 d=None, l=None):
        self.n = n #end-point abundances vector
        self.A = A #competition matrix
        self.C = C #resource preferences matrix
        self.D = D #crossfeeding matrix 
        self.r = r #resource growth rate
        self.d = d #species death rate
        self.l = l #leakage
        self.model = model #model on which the instance of the class is based
        self.presence = np.zeros(len(n), dtype = bool)
        ind_extant = np.where(n > 0)[0]
        self.presence[ind_extant] = True #indices of present species
        self.richness = len(ind_extant) #richness
        self.converged = True #did assembly process converge?
        self.t = np.array([0])
        self.abundances_t = n[:, np.newaxis]

    def assembly(self, tol=1e-9, delete_history=False, 
                 t_dynamics=False):
        ##GLV MODEL##
        if self.model.__name__ == 'GLV':
            if not t_dynamics:
                #integrate using lemke-howson algorithm
                n_eq = lemke_howson_wrapper(-self.A, self.r)
                #set to 0 extinct species
                ind_ext = np.where(n_eq < tol)[0]
                if any(ind_ext):
                    n_eq[ind_ext] = 0
                self.n = n_eq
                try:
                    self.presence[ind_ext] = False
                except:
                    import ipdb; ipdb.set_trace(context = 20)
                self.richness -= len(ind_ext) #update richness 
                #if it fails, use numerical integration
                if np.all(n_eq == 0):
                    t_dynamics = True
            else: 
                #integrate using numerical integration
                print('integrating system...')
                sol = prune_community(self.model, self.n,
                                      args=(self.A, self.r, tol), 
                                      events=single_extinction,
                                      t_dynamics=t_dynamics)
                if not sol:
                    self.converged = False
                    return self
                #store temporal dynamics
                self.t = cumulative_storing(self.t, sol.t, time = True)
                #make 1 dimensional again
                self.t = self.t[0]
                self.abundances_t = cumulative_storing(self.abundances_t, 
                                                        sol.y)
                self.n = sol.y[:, -1]
        else:
            print("model not available")
            raise ValueError
        return self

    def remove_spp(self, remove_ind):
        '''
        remove all species in vector 'remove_ind' from community
        '''
        if self.model.__name__ == 'GLV':
            #create a deep copy of comm to keep original unmodified
            new_comm = copy.deepcopy(self)
            #remove row and column indices 'remove_ind' from A
            del_row = np.delete(new_comm.A, remove_ind, axis=0)
            del_row_col = np.delete(del_row, remove_ind, axis=1)
            new_comm.A  = del_row_col
            #remove element from abundance and growth rate vectors
            new_comm.n = np.delete(new_comm.n, remove_ind)
            new_comm.r = np.delete(new_comm.r, remove_ind)
            #update presence vector
            new_comm.presence[remove_ind] = False
            #get number of species actually removed (i.e., only those whose 
            #abundance was different than 0)
            n_rem = sum(self.n[remove_ind]>0)
            #reduce richness accordingly
            new_comm.richness -= n_rem
        else:
            raise ValueError('unknown model name')
        return new_comm

    def add_spp(self, add_ind, **kwargs):
        '''
        add all the species in 'add_ind' which details are in **kwargs
        '''
        if self.model.__name__ == 'GLV':
            #create a deep copy of comm to keep original unmodified
            new_comm = copy.deepcopy(self)
            #switch to ones in the indices of introduced species
            new_comm.presence[add_ind] = True
            mask = new_comm.presence == True
            add_row = kwargs['row'][mask]
            add_col = kwargs['col'][mask]
            #map old index vector into new index vector
            new_add = index_mapping(add_ind, 
                                    np.where(new_comm.presence==False)[0])
            #update richness
            new_comm.richness += len(new_add)
            #delete diagonal element to adhere to previous dimensions of A
            add_row_d = np.delete(add_row, new_add)
            #add rows and columns at the end of matrix A
            new_comm.A = np.insert(new_comm.A, new_add, add_row_d, axis = 0)
            new_comm.A = np.insert(new_comm.A, new_add, 
                                   add_col.reshape(new_comm.richness, 
                                                   len(new_add)), axis = 1)
            #add element to growth rate
            new_comm.r = np.insert(new_comm.r, new_add, kwargs['r'])
            #update abundances
            new_comm.n = np.insert(new_comm.n, new_add, kwargs['x'])
        else:
            raise ValueError('unknown model name')
        return new_comm

    def is_subcomm(self, presence):
        '''
        determine if the presence/absence binary vector is a subset of the 
        community
        '''
        #CHECK IF THE VECTOR PRESENCE HAS TO BE BINARY OR IT CAN BE BOOLEAN
        set1 = set(np.where(self.presence == True)[0])
        set2 = set(np.where(presence == 1)[0])
        if set1 == set2:
            return False
        else: 
            return set1.issubset(set2)

    def delete_history(self):
        '''
        Delete history of assemlby, that is remove zeroed species, as well as
        absences from the presence vector
        '''
        #remove extinct species
        rem_ind = np.where(self.presence == 0)[0]
        comm = self.remove_spp(rem_ind)
        #remove from presence vector
        comm.presence = self.presence[self.presence]
        return comm

    def plotter(self):
        n_resources = 10
        n_consumers = 10
        t = self.t
        abundances = self.abundances_t
        for sp in range(n_resources + n_consumers):
            if sp<n_resources:
                plt.plot(t, abundances[sp])   
            else:
                plt.plot(t, abundances[sp], linestyle = '--')
        plt.xscale('log')
        plt.show()

###############################################################################

###############################################################################
#INTEGRATION
def lemke_howson_wrapper(A, r):
    np.savetxt('../data/A.csv', A, delimiter=',')
    np.savetxt('../data/r.csv', r, delimiter=',')
    os.system('Rscript call_lr.r')
    x = np.loadtxt('../data/equilibrium.csv', delimiter=',')
    #make sure I get an array-like object
    try: len(x)
    except: x = np.array([x])
    return x

def single_extinction(t, n, A, r, tol):
    n = n[n!=0]
    return np.any(abs(n) < tol) -1

def is_varying(sol_mat, tol):
    '''
    Check if all the solutions have reached steady state (constant)
    '''
    #Get differences between solutions
    diff_sol = sol_mat[:, 1:] - sol_mat[:, 0:-1]
    #Get last 3 timepoints
    last_3 = diff_sol[:, -1:-3:-1]
    #Note that we only impose three because there are no oscillations here. 
    varying = np.any(abs(last_3) > tol)
    return varying

def prune_community(fun, x0, args, events=(single_extinction, is_varying),  
                    t_dynamics=False):
    '''
    Function to prune community. Every time a species goes extinct, integration
    restarts with the pruned system
    '''
    #Extinctions don't triger end of integration, but reaching a constant
    #solution does
    single_extinction.terminal = False
    is_varying.terminal = True
    t_span = [0, 1e6]
    #add tolerance to tuple of arguments
    tol = args[-1]
    #get initial number of species
    n_sp = len(x0)
    varying = True
    #preallocate abundances and time vector
    ab_mat = x0[:, np.newaxis]
    t_vec = np.array([0])
    while varying:
        try:
            sol = solve_ivp(fun, t_span, x0, events=events, args=args, 
                            method='BDF') 
            #store times and abundances
            t_vec = cumulative_storing(t_vec, sol.t, time = True)
            ab_mat = cumulative_storing(ab_mat, sol.y)
            #update solution
            sol.t = t_vec
            sol.y = ab_mat
            #set species below threshold to 0
            end_point = sol.y[:, -1]
            ind_ext = np.where(end_point < tol)[0]
            end_point[ind_ext] = int(0)
            #update number of species
            n_sp = len(np.where(end_point>0)) 
            #initial condition of next integration is end point of previous one
            x0 = end_point
            #check if solution is constant
            varying = is_varying(sol.y, tol)
            print(sol.y[:, -1])
        except:
            sol = None
    return sol
###############################################################################

###############################################################################
#FUNCTIONS
def index_mapping(old_ind, del_ind):
    '''
    Given lists of indices of certain positions and deletions on a vector, 
    determine the new indices of positions once deletions are removed.
    Note that the intersection between old_ind and del_ind must be the empty
    set, and also that their union need not span the full length of the vector.

    Example:

        vector = np.array([1, 2, 3, 4, 5])
        old_index = [0, 3]
        del_index = [1, 4]
        new_index = index_mapping(old_index, del_index)
        print(vector[old_index])
        new_vector = np.delete(vector, del_index)
        print(new_vector[new_index])
        #the two print statements yield the same output
    '''
    return [i - sum([j < i for j in del_ind]) for i in old_ind]

def cumulative_storing(old_vector, new_vector, time = False):
    '''
    Concatenate two vectors, and in space and time (if a time vector)
    '''
    #make vectors have 2 axis 
    if old_vector.ndim < 2 and new_vector.ndim < 2: 
        old_vector = old_vector[np.newaxis,:]
        new_vector = new_vector[np.newaxis,:]
    elif old_vector.ndim == 2 and new_vector.ndim < 2:
        new_vector = new_vector[np.newaxis,:]
    if not np.any(old_vector):
        old_vector = new_vector
    else:
        vector_add = new_vector
        if time:
            vector_add = old_vector[:,-1] + new_vector
        old_vector = np.hstack((old_vector, vector_add[:,1:]))
    return old_vector

###############################################################################

