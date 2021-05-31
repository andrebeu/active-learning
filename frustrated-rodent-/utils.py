import numpy as np
import torch as tr
from collections import namedtuple
from torch.distributions.categorical import Categorical

BATCHSIZE = 1 # no batching, "online"



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
  
    def __init__(self,indim=2,nactions=3,stsize=24,gamma=1.0,learnrate=0.005,TDupdate=False):
        super().__init__()
        self.indim = indim
        self.stsize = stsize
        self.nactions = nactions
        self.learnrate = learnrate
        self.gamma = gamma
        self.TDupdate = TDupdate
        self.build()
        return None

    def build(self):
        """ concat input [s_tm1,a_tm1,r_tm1]
        """
        # policy parameters
        # self.rnncell = tr.nn.LSTMCell(self.indim,self.stsize,bias=True)
        self.rnn = tr.nn.LSTM(self.indim,self.stsize,bias=True)
        self.rnn_st0 = tr.nn.Parameter(tr.rand(2,1,self.stsize),requires_grad=True)
        self.rnn2val = tr.nn.Linear(self.stsize,1,bias=True)
        self.rnn2pi = tr.nn.Linear(self.stsize,self.nactions,bias=True)
        # optimization
        self.optiop = tr.optim.RMSprop(self.parameters(), 
          lr=self.learnrate
        )
        return None

    def reset_state(self):
        self.h_t,self.c_t = self.rnn_st0
        return None

    def unroll_trial(self,obsA):
        return rnn_states

    def unroll_trial_implicit(self,obsA):
        """
        given sequence of stimuli, unrolls rnn
         returns actions and hidden states
        efficient: forward layers applied in parallel
         """
        assert BATCHSIZE==1,'sqeueze batchdim'
        obsA = tr.Tensor(obsA).unsqueeze(1) # batchdim

        # propo rnn and forward pass head layers
        rnn_out,h_n = self.rnn(obsA)

        assert len(h_n) == 2,len(h_n)
        assert len(rnn_out) == len(obsA)
        vhatA = self.rnn2val(rnn_out)
        piact = self.rnn2pi(rnn_out)
        pism = piact.softmax(-1) 
        # probability of each action each timestep
        assert pism.shape == (len(obsA),1,self.nactions)
        """
        distribution constructor 
         takes probabilities of each category
        returns distribution object with
         .log_prob(action) and .sample() methods
        """
        pi_distr = Categorical(pism)
        return pi_distr,vhatA

    def update(self,expD):
        """ 
        supported REINFORCE and (broken) A2C updates 
        expects expD = {'state','obs','action','reward','pi','vhat'}
        """
        returns = tr.Tensor(compute_returns(expD['reward'],gamma=self.gamma))
        assert BATCHSIZE==1,'squeezing batchdim'
        vhats = tr.Tensor(expD['vhat']).squeeze()
        # form RL target
        if self.TDupdate: # actor-critic loss
            delta = tr.Tensor(expD['reward'][:-1])+self.gamma*vhats[1:].squeeze()-vhats[:-1].squeeze()
            delta = tr.cat([delta,tr.Tensor([0])])
        else: # REINFORCE
            delta = tr.Tensor(returns) - vhats
        # form RL loss
        logpr_actions = expD['logpr_actions']
        print('logpr_actions',tr.Tensor(logpr_actions).shape)
        print('delta',delta.shape)
        los_pi = tr.mean(delta*tr.Tensor(logpr_actions))
        ent_pi = tr.mean(tr.Tensor([distr.entropy().mean() for distr in expD['pi']])) # mean over time
        los_val = tr.mean(tr.square(returns - vhats)) # MSE
        los = 0.1*los_val-los_pi
        # update step
        self.optiop.zero_grad()
        los.backward()
        self.optiop.step()
        return None 



class PWMTaskFR():
    
    def __init__(self,stimdim=2,embed_stim='onehot',
        stim_set=[[0,1],[1,0]],stim_mean=None,stim_var=None,
        trlen=5
        ):
        """ 
        action space {0:hold,1:left,2:right}
        stim_set is list of int tuples [(Sa,Sb)]
        """
        self.max_reward = 12
        self.epoch_len = 60 # pub happens on 61
        self.trlen = trlen
        # fixed
        self.stim_set = stim_set
        self.action_set = [0,1,2]
        self.stimdim = stimdim
        # embedding
        if embed_stim=='onehot':
            self.embed_stim = self._embed_onehot
        elif embed_stim=='gauss':
            self.embed_stim = lambda X: self._embed_gauss(X,stim_mean,stim_var)
        return None
    
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
        if ttype==False: # ITI
            obsA = np.zeros([trlen,self.stimdim]) # two stim
            stateL = np.zeros(trlen)
            return stateL,obsA
        delay = trlen-2
        assert delay>0 # 2 stim
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
        stateL[-1] = 1+int(SAi>SBi) 
        return stateL,obsA

    def _reward_lastaction_only(self,stateL,actionL):
        """
        reward 0 hold 
        reward +1 action
        """
        reward_action = int(stateL[-1] == actionL[-1])
        print('ALshape',actionL.shape)
        reward = np.concatenate([np.zeros_like(actionL[:-1]),[reward_action]])
        assert reward.shape == stateL.shape
        return reward

    def reward_fn(self,stateL,actionL):
        """ 
        computes reward and determines 
        whether next trial is valid or timeout
        """
        # hold everywhere except action
        valid_trial = np.all(actionL[:-1].numpy() == 0)
        if valid_trial:
            rewardL = self._reward_lastaction_only(stateL,actionL)
        else:
            rewardL = np.zeros(len(stateL))
        return rewardL,valid_trial


    def pub(self,epoch_data):
        return epoch_data
    
    def padding(self,epoch_data):
        return epoch_data


def run_epoch_FR(agent,task):
    epoch_data = {
        'state':[],
        'obs':[],
        'action':[],
        'reward':[],
        'vhat':[],
        'pi':[],
        'ttype':[],
        'logpr_actions':[]

    } 
    valid_trial = True
    # loop over trials within epoch
    trlen = task.trlen
    epoch_len = task.epoch_len
    # epoch_len = trlen
    tr_c = 0
    while tr_c+trlen <= epoch_len:
        tr_c += trlen
        epoch_data['ttype'].append(int(valid_trial))
        # run trial
        stateL,obsA = task.sample_trial(valid_trial)
        """ 
        currently unroll_trial uses RNN
        between trials RNN gets reset
        leads to issues in gradient computatio
        TODO: switch to explicit rnn unroll
         apply pihead within here to sample action
         save softmax act of pi head over trials
          then make distr obj in update
          parallelizing over all time steps
          to call .log_prob(actions)
        """
        pi_distr,vhatA = agent.unroll_trial(obsA)
        actionL = pi_distr.sample()
        logpr_actions = pi_distr.log_prob(actionL)
        ##
        rewardL,valid_trial = task.reward_fn(stateL,actionL)
        # record
        epoch_data['state'].extend(stateL)
        epoch_data['obs'].extend(obsA)
        epoch_data['action'].extend(actionL)
        epoch_data['reward'].extend(rewardL)
        epoch_data['vhat'].extend(vhatA)
        epoch_data['logpr_actions'].extend(logpr_actions)
        epoch_data['pi'].append(pi_distr)
    ## padding and pub modify trial data
    epoch_data = task.padding(epoch_data) 
    epoch_data = task.pub(epoch_data) 
    return epoch_data


def calc_epoch_pi_los(expD):
    """ 
    helper for calculating loss when different 

    unpack dataD (dict of lists of tensors)
        lists correspond to trials in epoch
    into expD (dict of tensors)
        assumed by agent.update(expD)
    """
    data = {}
    ntrials = len(data['reward'])
    print('ntr',ntrials)
    return expD


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
        assert len(epoch_data['reward'])==task.epoch_len
        agent.update(epoch_data)
