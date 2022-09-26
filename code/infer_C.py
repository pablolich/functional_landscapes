#!/usr/bin/env python3

__appname__ = '[App_name_here]'
__author__ = 'Pablo Lechon (plechon@ucm.es)'
__version__ = '0.0.1'

## IMPORTS ##

import sys
import numpy as np
import pandas as pd
from essential_tools import *

## CONSTANTS ##


## FUNCTIONS ##

def main(argv):
    '''Main function'''
    #Set parameters
    n, m = (3, 3)
    d = np.repeat(np.random.uniform(0, 1), n)
    r = 1+max(d)+np.random.uniform(0, 1, m)
    #1. Generate simulated data maybe add some errors all the monos, each with
    #all the resources
    #2. Propose a C
    #3. Build corresponding A
    #4. Compute z with the proposed A
    #5. Compute P^T = Az^T
    #6. Find an updated A^-1 by numerically minimizing ||\sim z^T - A^-1P^T||
    #7. Invert A^-1 to recover A.
    #8. Repeat 3-7 until convergence

    return 0

## CODE ##

if (__name__ == '__main__'):
    status = main(sys.argv)
    sys.exit(status)
     

