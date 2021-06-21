import numpy as np
import torch as tr
from collections import namedtuple
from torch.distributions.categorical import Categorical
import multiprocessing as mp
import concurrent


BATCHSIZE = 1 # no batching, "online"
SDIM = 6 # pwm5


def compute_returns(rewards,gamma=1.0):
    """ 
    given rewards, compute discounted return
    G_t = sum_k [g^k * r(t+k)]; k=0...T-t
    """  
    assert BATCHSIZE==1,'return computation'
    T = len(rewards) 
    returns = np.array([
        np.sum(np.array(
            rewards[t:])*np.array([
                gamma**i for i in range(T-t)]
        )) for t in range(T)
    ]) ## not sure how to parse this
    return returns


def process_epdata(epdata):
    """ 
    given epoch data from actor (run_epoch fun)
    process for agent.update method
    """
    epdata = flattenDoLoL(epdata)
    list_of_cat = ['pism','vhat','logpr_actions']
    for k in list_of_cat:
        epdata[k] = tr.cat(epdata[k]).squeeze()
    return epdata


def flattenDoLoL(D):
    """ dict of list of lists flattened into dict of lists"""
    exceptL = ['distr'] # exception
    return {k:[i for s in v for i in s] for k,v in D.items() if k not in exceptL}


# runtime


def run_epoch_FR(agent,task,pub=False,vto=True):
    """ FRsim env-actor logic 
    pub: `bool`, include pub reward
    vto: `bool`, allow violation timeout trial
    """
    epoch_data = {
        'state':[],
        'obs':[],
        'action':[],
        'reward':[],
        'vhat':[],
        'distr':[],
        'ttype':[],
        'logpr_actions':[],
        'pism':[],
    } 
    trlen = task.trlen
    epoch_len = task.epoch_len
    # inital trial settings
    tr_c = 0
    agent.reset_rnn_state()
    valid_trial = True
    while tr_c+trlen <= epoch_len:
        # agent.reset_rnn_state()
        # print('tr',tr_c)
        tr_c += trlen
        epoch_data['ttype'].append([int(valid_trial)])
        # run trial
        stateL,obsA = task.sample_trial(valid_trial)
        # fw agent
        vhatL,pism,pi_distr,actionL,logpr_actions = agent.play_trial(obsA)
        rewardL,valid_trial = task.reward_fn(stateL,actionL)
        if not vto: valid_trial = True
        # record
        epoch_data['state'].append(stateL)
        epoch_data['obs'].append(obsA)
        epoch_data['action'].append(actionL)
        epoch_data['reward'].append(rewardL)
        epoch_data['vhat'].append(vhatL)
        epoch_data['logpr_actions'].append(logpr_actions)
        epoch_data['distr'].append(pi_distr)
        epoch_data['pism'].append(pism)
    # padding and pub modify trial data
    # epoch_data = task.padding(epoch_data) 
    if pub:
        epoch_data = task.pub(agent,epoch_data) 
    return epoch_data


def seed_exp(seed,args):
    """ loss [(value, policy),neps] """
    # setup
    np.random.seed(seed)
    neps = args['train']['neps']
    # task and agent definition
    agent = ActorCritic(**args['agent'])
    task = PWMTaskFR(**args['task'])
    # init loop vars
    reward = -np.ones(neps)
    loss = -np.ones([2,neps]) 
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


# multiprocess fun for parallelizing simulations across seeds
def exp_mp(seed_exp,nseeds,gsvar=None):
    """ 
    first argument is seed_exp method
    seed_exp takes one dummy argument `seed_num`
     placeholder for iterating over seeds
    seed_exp could also take second argument `gsvar`
     variable condition repeated over seeds 
     used for gridsearching
    returns list of outputs from each seed_exp
    """
    with concurrent.futures.ProcessPoolExecutor() as exe:
        data = exe.map(seed_exp, np.arange(nseeds), np.repeat(gsvar,nseeds))
    return np.array([i for i in data])
  

# used in nb
def plt_metrics(data,condL,mawin=10):
  """ plot data{reward,vloss,ploss}
  condL refers to gridsearched variable values
  input shapes [num_cond,seed,epochs]
  moving average to smooth plots
  condL labels conds along num_cond
  """
  _,ncond,nseeds,nepochs = data.shape
  mdata = data.mean(2)
  sdata = data.std(2)/np.sqrt(nseeds)
  # plot setup
  f,axar = plt.subplots(ncond,3,figsize=(35,6*ncond),sharex=True)
  axar[0,0].set_title('reward')
  axar[0,1].set_title('value loss')
  axar[0,2].set_title('policy loss')
  # loop over axes
  for ci in range(ncond):
    axa = axar[ci]
    axa[0].set_ylabel(condL[ci])
    axa[0].set_ylim(0.2,1)
    for ii in range(3):
      ax = axa[ii]
      # moving average
      M = mdata[ii,ci].reshape(-1,mawin).mean(-1)
      S = sdata[ii,ci].reshape(-1,mawin).mean(-1)
      ax.plot(M)
      ax.fill_between(range(len(M)),M-S,M+S,alpha=0.2)

  return None