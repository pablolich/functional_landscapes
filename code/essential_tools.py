###############################################################################
#IMPORTS
import copy
import os
import numpy as np
from scipy.integrate import solve_ivp
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

def CR(t, x, C, r, d, tol):
    return 0

def CR_crossfeeding(t, x, C, D, r, d, l, tol):
    n = len(d)
    m = len(r)
    I = np.identity(n)
    xdot = I@x@(-d + (1-l)*C.T@y)
    ydot = I@x@(r - y + (l*D-I)@C@x)
    return (list(xdot.reshape(n))+list(ydot.reshape(m)))
###############################################################################

###############################################################################
#CLASSES
class Community:
    def __init__(self, n, model, A=None, C=None, D=None, r=None, rho=None, 
                 d=None, l=None):
        self.n = n #abundance vector
        self.A = A #competition matrix
        self.C = C #resource preferences matrix
        self.D = D #crossfeeding matrix 
        self.r = r #growth rate
        self.d = d #death rate
        self.l = l #leakage
        self.model = model #model on which the instance of the class is based
        self.presence = np.zeros(len(n), dtype = bool)
        ind_extant = np.where(n > 0)[0]
        self.presence[ind_extant] = True #indices of present species
        self.richness = len(ind_extant) #richness
        self.converged = True #did assembly process converge?

    def assembly(self, tol=1e-9, delete_history=False):
        if self.model.__name__ == 'GLV':
            #integrate using lemke-howson algorithm
            n_eq = lemke_howson_wrapper(-self.A, self.r)
            if np.all(n_eq == 0):
                #lemke-howson got stuck, integrate using differential equations
                print('integrating dynamics the hard way')
                sol = prune_community(self.model, self.n, tol,
                                      args=(self.A, self.r), 
                                      events=single_extinction)
                if sol:
                    n_eq = sol.y[:, -1]
                else:
                    self.converged = False
                    return self
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
        elif self.model.__name__ == 'CR_crossfeeding':
            #numerically integrate differential equations
            sol = prune_community(self.model, self.n, tol, 
                                  args=(self.C, self.D, self.r, self.d,self.l),
                                  events=single_extinction)
            n_eq = sol.y[:, -1]
            #set to 0 extinct species
            ind_ext = np.where(n_eq < tol)[0]
            if any(ind_ext):
                n_eq[ind_ext] = 0
            #update rest of the attributes
            self.n = n_eq
            self.presence[ind_ext] = False
            self.richness -= len(ind_ext) #update richness 

        else:
            print("haven't coded up other type of models yet")
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

class Environment:
    def __init__(self, r):
        self.r = r #supply rate of each resource
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

def check_constant(sol_mat, tol):
    '''
    Check if all the solutions have reached steady state (constant)
    '''
    #Get differences between solutions
    diff_sol = sol_mat[:, 1:] - sol_mat[:, 0:-1]
    #Get last 3 timepoints
    last_3 = diff_sol[:, -1:-3:-1]
    #Note that we only impose three because there are no oscillations here. 
    const = np.all(abs(last_3) < tol)
    return const

def prune_community(fun, x0, tol, args, events=single_extinction):
    '''
    Function to prune community. Every time a species goes extinct, integration
    restarts with the pruned system
    '''
    single_extinction.terminal = True
    t_span = [0, 1e6]
    #add tolerance to tuple of arguments
    args += (tol, )
    #get initial number of species
    n_sp = len(x0)
    constant = False
    while n_sp > 1 or not constant :
        try:
            sol = solve_ivp(fun, t_span, x0, events=events, args=args, 
                            method='BDF') 
            #set species below threshold to 0
            end_point = sol.y[:, -1]
            ind_ext = np.where(end_point < tol)[0]
            end_point[ind_ext] = int(0)
            n_sp = len(end_point) - len(ind_ext)
            #initial condition of next integration is end point of previous one
            x0 = end_point
            #check if solution is constant
            constant = check_constant(sol.y, tol)
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
###############################################################################

