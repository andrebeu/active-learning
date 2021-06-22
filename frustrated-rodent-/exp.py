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

# parameter passing
param_str = str(sys.argv[1])
p1,p2,p3,p4 = param_str.split()
lrate = float(p1) # 0.005
stsize = float(p2) # 48
vlos = float(p3) # 8
elos = float(p4) # 0


# parameter dict setup
nseeds = 3
## setup
args = {
  'train':{
    'neps':200000
  },
  'agent':{
    'gamma':1.0,
    'learnrate':lrate,
    'stsize':stsize,
    'vlos_w':vlos,
    'elos_w':elos
  },
  'task':{
    'stimset':'pwm5',
    'epoch_len':9, ## 3 trials len 3 each
    'trlen':3
  }
}
mtag = "-".join(["-".join(["%s_%s"%(i,j) for i,j in v.items()]) for k,v in args.items()])
mtag += "-tstamp_%s"%tstamp
print('model tag =',mtag)

def seed_exp(seed,args):
    """ loss [(value, policy),neps] """
    # setup
    np.random.seed(seed)
    tr.manual_seed(seed)
    neps = args['train']['neps']
    # task and agent definition
    agent = ActorCritic(**args['agent'])
    task = PWMTaskFR(**args['task'])
    # init loop vars
    reward = -np.ones(neps)
    loss = -np.ones([3,neps]) 
    pism = -np.ones([3,neps])
    trcount = -np.ones(neps)
    L = []
    # loop over epochs
    for epoch in range(neps):
        # run
        epoch_data = run_epoch_FR(agent,task)
        epoch_data = process_epdata(epoch_data)
        update_data = agent.update(epoch_data)
        # record
        trcount[epoch] = np.sum(epoch_data['ttype'])
        reward[epoch] = np.sum(epoch_data['reward'])/task.ntrials
        loss[:,epoch] = list(update_data.values())
    data = {
        'loss':loss,
        'reward':reward,
        'trcount':trcount
    }
    return data

# run multiseed exp
dataL = exp_mp(seed_exp,nseeds=nseeds,gsvar=args)
# unpack data
loss = np.array([d['loss'] for d in dataL])
vloss,ploss,eloss = loss.transpose(1,0,2)
reward  = np.array([d['reward'] for d in dataL])
trcount = np.array([d['trcount'] for d in dataL])

## plots
MAw = 10

data = np.array([reward,vloss,ploss,eloss,trcount])
labL = ['reward','vloss','ploss','eloss','trcount']
for idx in range(len(data)):
    ## plot trial counts
    plt.figure(figsize=(15,8))
    plt.plot(data[idx].mean(0).reshape(-1,MAw).mean(-1),)
    plt.savefig('figures/%s-%s.jpg'%(labL[idx],mtag))
    plt.close('all')

print('~~exp.py finished~~')