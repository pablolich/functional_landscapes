#Wrapper to call L-H.R from python
setwd("~/Desktop/functional_landscapes/code/")
source('L-H.R')
#load parameters
A = data.matrix(read.csv('../data/A.csv', header = F))
r = data.matrix(read.csv('../data/r.csv', header = F))
x = get_final_composition(A,  r)
#Transform NaN and infs to 0
ind_nan = which(is.nan(x))
ind_inf = which(is.infinite(x))
x[ind_nan] = 0
x[ind_inf] = 0
#save results to be used in python
write.table(x, '../data/equilibrium.csv', row.names = F, col.names = F)
