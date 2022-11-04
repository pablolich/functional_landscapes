#!/usr/bin/env python3

__appname__ = '[App_name_here]'
__author__ = 'Pablo Lechon (plechon@ucm.es)'
__version__ = '0.0.1'

## IMPORTS ##

import sys
import numpy as np
import pandas as pd
import os
home_dir = os.path.expanduser('~')
sys.path.append(home_dir+'/Desktop/usefulprograms/code')
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

def main(argv):
    '''Main function'''
    n, m = (30, 30)
    C = np.random.uniform(0, 1, size=(m,n))
    r = np.random.uniform(1, 2, n)
    d = np.random.uniform(0, 1, n)
    rho = np.concatenate((r, -d))
    l=0
    D = np.zeros((m, m))
    A = C2A(C, D, l)
    #create community and assemble without timescale separation
    comm = Community(np.ones(n+m), GLV, A=A, rho=rho).assembly()
    #perform time-scale separation
    A_s = -C@C.T
    rho_s = C@r-d
    #assemble again
    comm_s = Community(np.ones(n), GLV, A=A_s, rho=rho_s).assembly()
    plt.scatter(comm.n[m:], comm_s.n)
    plt.show()
    #remove extinctions from original
    subcomm = comm.remove_spp(np.where(comm.presence==False)[0])
    m_s = len(np.where(comm.n[0:m]>0)[0])
    n_s = len(np.where(comm.n[m:]>0)[0])
    C_sub = A2C(subcomm.A, m_s, n_s, l)
    r_sub = r[np.where(comm.presence[0:m]==True)[0]]
    d_sub = d[np.where(comm.presence[m:]==True)[0]]
    #new matrix of interactions for time scale separation of the pruned one
    A_s_sub = -C_sub@C_sub.T
    rho_s_sub = C_sub@r_sub-d_sub
    #integrate
    subcomm_s = Community(np.ones(n_s), GLV, A=A_s_sub, rho=rho_s_sub)
    subcomm_s.assembly()


    return 0

## CODE ##

if (__name__ == '__main__'):
    status = main(sys.argv)
    sys.exit(status)
     

