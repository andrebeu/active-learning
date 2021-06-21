import sys
import os
import itertools
import numpy as np
import torch as tr
from matplotlib import pyplot as plt
import multiprocessing as mp
import concurrent
import seaborn as sns
sns.set_context('talk')
import time
tstamp = time.perf_counter_ns()
#
from utils import *
from model import *


## setup
nseeds,neps = 3,200000
args = {
  'train':{
    'neps':neps
  },
  'agent':{
    'gamma':1.0,
    'learnrate':0.005,
    'lweight':8 # gridsearching
  },
  'task':{
    'stimset':'pwm5',
    'epoch_len':9, ## 3 trials len 3 each
    'trlen':3
  }
}


# run multiseed exp
dataL = exp_mp(seed_exp,nseeds=nseeds,gsvar=args)

# unpack data
loss = np.array([d['loss'] for d in dataL])
reward  = np.array([d['reward'] for d in dataL])
trcount = np.array([d['trcount'] for d in dataL])

# ### plot trial counts

plt.figure(figsize=(15,8))
# trial count
for idx in range(len(lwL)): # loop over conds
  plt.plot(trcount[idx].mean(0).reshape(-1,20).mean(-1),
          label=lwL[idx])
plt.ylabel('trial_count')
plt.legend()

# plt.savefig
plt.close('all')

## plot reward and loss

# reshape data for plotting
vloss = loss[:,:,0,:]
ploss = loss[:,:,1,:]
data = np.array([reward,vloss,ploss])
data.shape # [{R/vL/pL},cond,seeds,epochs]

plt_metrics(data,lwL,mawin=10)
# plt.savefig
plt.close('all')

