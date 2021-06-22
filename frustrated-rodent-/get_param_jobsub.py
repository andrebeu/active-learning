import sys
import itertools
import numpy as np

"""
controls the gridsearch parameter space
given an index, return parameter string
""" 


param_set_idx = int(sys.argv[1])

p1L = [0.0005,0.001,0.005] # lrate
p2L = [24,48] # stsize 
p3L = [2,8] # vlos 
p4L = [0,0.05] # elos

print('nconds',len(p1L)*len(p2L)*len(p3L)*len(p4L))

itrprod = itertools.product(p1L,p2L,p3L,p4L)

for idx,paramL in enumerate(itrprod):
  if idx == param_set_idx:
    print(" ".join([str(i) for i in paramL]))
    break


