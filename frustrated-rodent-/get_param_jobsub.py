import sys
import itertools
import numpy as np

"""
controls the gridsearch parameter space
given an index, return parameter string
""" 


param_set_idx = int(sys.argv[1])

p1L = np.arange(0.02,0.041,0.005)
p2L = np.arange(0.8,1.11,0.1)
p3L = np.arange(0.25,0.451,0.05)
p4L = np.arange(0.05,0.251,0.05)

print('nconds',len(p1L)*len(p2L)*len(p3L)*len(p4L))


itrprod = itertools.product(p1L,p2L,p3L,p4L)

for idx,paramL in enumerate(itrprod):
  if idx == param_set_idx:
    print(" ".join([str(i) for i in paramL]))
    break


