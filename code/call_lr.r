#Wrapper to call L-H.R from python
#setwd("~/Desktop/cr_doomsday/code")
source('L-H.R')
#load parameters
A = data.matrix(read.csv('../data/A.csv', header = F))
r = data.matrix(read.csv('../data/r.csv', header = F))
x = get_final_composition(A,  r)
#save results to be used in python
write.table(x, '../data/equilibrium.csv', row.names = F, col.names = F)
