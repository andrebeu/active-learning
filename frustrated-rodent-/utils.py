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



class ActorCritic(tr.nn.Module):
    """ implementation NOTEs: 
    """
  
    def __init__(self,indim=SDIM,nactions=3,stsize=48,gamma=1.0,
        learnrate=0.005,TDupdate=False,lweight=0.1):
        super().__init__()
        self.indim = indim
        self.stsize = stsize
        self.nactions = nactions
        self.learnrate = learnrate
        self.gamma = gamma
        self.TDupdate = TDupdate
        self.vlos_weight = lweight
        self.build()
        self.reset_rnn_state()
        return None

    def build(self):
        """ concat input [s_tm1,a_tm1,r_tm1]
        """
        # policy parameters
        # self.rnncell = tr.nn.LSTMCell(self.indim,self.stsize,bias=True)
        self.rnn = tr.nn.LSTM(self.indim,self.stsize,bias=True)
        self.rnncell = tr.nn.LSTMCell(self.indim,self.stsize,bias=True)
        self.rnn_st0 = tr.nn.Parameter(tr.rand(2,1,1,self.stsize),requires_grad=True)
        self.rnn2val = tr.nn.Linear(self.stsize,1,bias=True)
        self.rnn2pi = tr.nn.Linear(self.stsize,self.nactions,bias=True)
        # optimization
        self.optiop = tr.optim.RMSprop(self.parameters(), 
          lr=self.learnrate
        )
        return None

    def reset_rnn_state(self):
        self.h_t,self.c_t = self.rnn_st0
        return None

    def unroll_trial(self,obsA):
        """ 
        useful for analyzing within trial 
        model representations 
        """
        assert BATCHSIZE==1,'sqeueze batchdim'
        obsA = tr.Tensor(obsA).unsqueeze(1) # batchdim
        h_t,c_t = self.h_t,self.c_t
        rnn_hstL = []
        for obs in obsA:
            h_t,c_t = self.rnncell(obs,(h_t,c_t))
            rnn_hstL.append(h_t)
        rnn_out = tr.stack(rnn_hstL) # stack 
        return rnn_out

    def unroll_rnn(self,obsA):
        """
        given sequence of stimuli [steps,stimdim], unrolls rnn
         returns actions and hidden states
        efficient: forward layers applied in parallel
         """
        assert BATCHSIZE==1,'sqeueze batchdim'
        obsA = tr.Tensor(obsA).unsqueeze(1) # batchdim
        # prop rnn and forward pass head layers
        rnn_out,(self.h_t,self.c_t) = self.rnn(obsA,(self.h_t,self.c_t))
        return rnn_out
    
    def play_trial(self,obsA):
        rnn_out = self.unroll_rnn(obsA)
        vhatL = self.rnn2val(rnn_out)
        piact = self.rnn2pi(rnn_out)
        pism = piact.softmax(-1) 
        pi_distr = Categorical(pism)
        actionL = pi_distr.sample()
        logpr_actions = pi_distr.log_prob(actionL)
        return vhatL,pism,pi_distr,actionL,logpr_actions

    def compute_TDdelta(self,rewards,vhats):
        assert BATCHSIZE==1,'squeezing batchdim'
        # form RL target
        Ghat = tr.Tensor(expD['reward'][:-1]) + self.gamma*vhats[1:].squeeze()
        delta = Ghat - vhats[:-1].squeeze()
        delta = tr.cat([delta,tr.Tensor([0])])
        return delta

    def update(self,expD):
        """ 
        supported REINFORCE and (broken) A2C updates 
        expects expD = {'state','obs','action','reward','pi','vhat'}
        expects processed shapes and types
        """
        ## unpack expD
        rewards = expD['reward'] # 1D
        vhats = expD['vhat'] # 1D
        logpr_actions = expD['logpr_actions'] # 1D
        ##  
        returns = tr.Tensor(compute_returns(rewards,self.gamma))
        delta = returns - vhats
        # form RL loss
        los_pi = tr.sum(delta*logpr_actions)
        los_val = tr.mean(tr.square(delta)) # MSE
        # ent_pi = tr.mean(tr.Tensor([distr.entropy().mean() for distr in expD['pi']])) # mean over time
        # update step
        self.optiop.zero_grad()
        los = (self.vlos_weight*los_val)-los_pi
        los.backward()
        self.optiop.step()
        # return obj
        update_data = {'vlos':los_val,'plos':los_pi}
        return update_data


class PWMTaskFR():
    
    def __init__(self,stimdim=SDIM,embed_stim='onehot',
        stimset='pwm0',stim_mean=None,stim_var=None,
        trlen=3,epoch_len=60
        ):
        """ 
        action space {0:hold,1:left,2:right}
        stim_set is list of int tuples [(Sa,Sb)]
        """
        self.epoch_len = epoch_len # pub happens on 61
        self.trlen = trlen
        self.ntrials = self.epoch_len//trlen
        self.max_trial_reward = self.ntrials
        # fixed
        self.stim_set = self._get_stimset(stimset)
        self.action_set = [0,1,2]
        self.stimdim = stimdim
        # embedding
        if embed_stim=='onehot':
            self.embed_stim = self._embed_onehot
        elif embed_stim=='gauss':
            self.embed_stim = lambda X: self._embed_gauss(X,stim_mean,stim_var)
        return None
    
    def _get_stimset(self,setstr):
        if setstr == 'pwm0':
            stim_set=[[0,1],[1,0]]
        elif setstr == 'pwm5':
            stim_set=[
                [0,1],[1,0],
                [1,2],[2,1],
                [2,3],[3,2],
                [3,4],[4,3],
                ]
        return stim_set

    def _embed_onehot(self,intL):
        E = np.eye(self.stimdim)
        return E[intL]

    def _embed_gauss(self,intL,meanL,var):
        """ 
        intL is index of stimuli to embed
        meanL is list of means of stimuli
        assumes same var 
        E is [stimdim,nstim]
        """
        E = np.random.normal(meanL,var,size=[self.stimdim,len(meanL)])
        return E.T[intL]

    def sample_trial(self,ttype):
        """ 
        returns stateL (trlen) and obsA (trlen,stimdim)
        state indicates rewarded action
        - action_t = actors(obs)
        - reward_t = reward_fn(state,action)
        """
        trlen = self.trlen
        if int(ttype)==False: # ITI
            obsA = np.zeros([trlen,self.stimdim]) # two stim
            stateL = np.zeros(trlen)
            return stateL,obsA
        elif int(ttype)==2: # PUB trial
            None
        delay = trlen-2
        assert delay>=0 # 2 stim
        ## NB stimulus selection assumes batchsize1
        assert BATCHSIZE==1, 'sampling one stimset'
        SAi,SBi = self.stim_set[np.random.choice(len(self.stim_set))]
        # embed stim
        SA,SB = self.embed_stim([SAi,SBi])
        obsA = np.zeros([trlen,self.stimdim]) # two stim
        obsA[0] = SA
        obsA[-1] = SB
        # instructs rewarded action 
        stateL = np.zeros(trlen)
        stateL[-1] = 1+int(SAi<SBi) 
        return stateL,obsA

    def _reward_lastaction_only(self,stateL,actionL):
        """
        reward 0 hold 
        reward +1 action
        """
        reward_action = int(stateL[-1] == actionL[-1])
        reward = np.concatenate([np.zeros(len(actionL[:-1])),[reward_action]])
        assert reward.shape == stateL.shape
        return reward
    
    def _reward_invalid_trial(self,stateL,actionL):
        return None

    def reward_fn(self,stateL,actionL):
        """ 
        given states [trlen] and actions [trlen] 
         computes trial reward and determines 
         whether next trial is valid or timeout
          if previous trial is timeout
          rewards are all zero and next trial is valid
        """
        if stateL[-1] == 0: # current trial is invalid
            next_trial_is_valid = True
            rewardL = tr.zeros(len(stateL))
        else: # current trial is valid
            # hold everywhere except action
            next_trial_is_valid = np.all(actionL[:-1].numpy() == 0)
            rewardL = self._reward_lastaction_only(stateL,actionL)
        return tr.Tensor(rewardL),next_trial_is_valid
    

    def pub(self,agent,epoch_data):
        """ pub """
        assert False, 'nopub'
        pub_state = 9
        pub_obs_int = SDIM-1 # last obs vector
        Spub = self.embed_stim(pub_obs_int)
        ## fw agent
        vhatL,pism,pi_distr,actionL,logpr_actions = agent.play_trial([Spub])
        trial_rewards = tr.sum(tr.cat(epoch_data['reward']))
        pub_reward = [self.ntrials - trial_rewards]
        ##
        epoch_data['state'].append([pub_state])
        epoch_data['obs'].append([Spub])
        epoch_data['action'].append(actionL)
        epoch_data['reward'].append(pub_reward)
        epoch_data['vhat'].append(vhatL)
        epoch_data['logpr_actions'].append(logpr_actions)
        epoch_data['distr'].append(pi_distr)
        epoch_data['pism'].append(pism)
        return epoch_data



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
  

if __name__ == "__main__":
    agent_kw = {}
    agent = ActorCritic(**agent_kw)
    for epoch in range(2):
        print('ep',epoch,)
        task_kw = {'trlen':5}
        task = PWMTaskFR(**task_kw)
        epoch_data = run_epoch_FR(agent,task) # [epochlen]
        print(
            len(epoch_data['reward']),
            len(epoch_data['ttype'])
        )
        agent.update(epoch_data)
