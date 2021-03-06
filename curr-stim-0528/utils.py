import numpy as np
import torch as tr
from collections import namedtuple
from torch.distributions.categorical import Categorical



BATCHSIZE = 1 # no batching, "online"


class Env():

    def __init__(self,actor,task):
        self.actor = actor
        self.task = task
        return None

    def reward_hold_and_lastaction(self,stateL,obsL,actionL):
        """
        reward +1 hold 
        reward +1 action
        """
        reward_hold = np.equal(actionL[:-1],np.zeros_like(actionL[:-1]))
        reward_action = int(stateL[-1] == actionL[-1])
        assert BATCHSIZE == 1, 'squeezing batchsize'
        reward = np.concatenate([reward_hold.squeeze(),[reward_action]])
        assert reward.shape == stateL.shape
        return reward

    def reward_action_punish_nohold(self,stateL,obsL,actionL):
        """ 
        punish -1 nohold
        reward +1 action
        """
        punish_hold = -np.not_equal(actionL[:-1],np.zeros_like(actionL[:-1]))
        reward_action = int(stateL[-1] == actionL[-1])
        assert BATCHSIZE == 1, 'squeezing batchsize'
        reward = np.concatenate([punish_hold.squeeze(),[reward_action]])
        assert reward.shape == stateL.shape
        return reward 

    def reward_custom(self,stateL,obsL,actionL,holdR=1,actionR=1):
        """ 
        punish -1 nohold
        reward +1 action
        """
        reward_hold = holdR*np.equal(actionL[:-1],np.zeros_like(actionL[:-1]))
        reward_action = actionR*int(stateL[-1] == actionL[-1])
        assert BATCHSIZE == 1, 'squeezing batchsize'
        reward = np.concatenate([reward_hold.squeeze(),[reward_action]])
        assert reward.shape == stateL.shape
        return reward 

    def reward_fn(self,stateL,obsL,actionL):
        return self.reward_hold_and_lastaction(stateL,obsL,actionL)

    # used in delay 
    def run_pwm_trial(self,delay,update=True):
        """ 
        sample trial, unroll agent, compute rewards
         perform model update
        used for training and eval
        """
        stateL,obsA = self.task.sample_trial(delay+2)
        pi_distr,vhatA = self.actor.unroll_trial(obsA)
        assert BATCHSIZE == 1, 'use .sample([b])'
        actionL = pi_distr.sample()
        rewardL = self.reward_fn(stateL,obsA,actionL)
        assert len(actionL)==len(obsA)==len(rewardL)
        # 
        trial_data = {
            'state':stateL,
            'obs':obsA,
            'action':actionL,
            'reward':rewardL,
            'vhat':vhatA,
            'pi':pi_distr
        }
        if update:
            self.actor.update(trial_data)
        return trial_data
    
    # used in stimvar
    def run_pwm_trial2(self,update=True):
        """ 
        sample trial, unroll agent, compute rewards
         perform model update
        used for training and eval
        """
        stateL,obsA = self.task.sample_trial()
        pi_distr,vhatA = self.actor.unroll_trial(obsA)
        assert BATCHSIZE == 1, 'use .sample([b])'
        actionL = pi_distr.sample()
        rewardL = self.reward_fn(stateL,obsA,actionL)
        assert len(actionL)==len(obsA)==len(rewardL)
        # 
        trial_data = {
            'state':stateL,
            'obs':obsA,
            'action':actionL,
            'reward':rewardL,
            'vhat':vhatA,
            'pi':pi_distr
        }
        if update:
            self.actor.update(trial_data)
        return trial_data

class PWMTask():
    
    def __init__(self,stim_set,stimdim=2,embed_stim='onehot',
        stim_mean=None,stim_var=None):
        """ 
        action space {0:hold,1:left,2:right}
        stim_set is list of int tuples [(Sa,Sb)]
        """
        if embed_stim=='onehot':
            self.embed_stim = self.embed_onehot
        elif embed_stim=='gauss':
            self.embed_stim = lambda X: self.embed_gauss(X,stim_mean,stim_var)
        self.stim_set = stim_set
        self.action_set = [0,1,2]
        self.stimdim = stimdim
        return None

    def sample_trial(self,trlen=5):
        """ 
        returns stateL (trlen) and obsA (trlen,stimdim)
        state indicates rewarded action
        - action_t = actors(obs)
        - reward_t = reward_fn(state,action)
        """
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

    def embed_onehot(self,intL):
        E = np.eye(self.stimdim)
        return E[intL]

    def embed_gauss(self,intL,meanL,var):
        """ 
        intL is index of stimuli to embed
        meanL is list of means of stimuli
        assumes same var 
        E is [stimdim,nstim]
        """
        E = np.random.normal(meanL,var,size=[self.stimdim,len(meanL)])
        return E.T[intL]



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
        returns = tr.Tensor(compute_returns(expD['reward'],gamma=self.gamma) )
        assert BATCHSIZE==1,'squeezing batchdim'
        vhats = expD['vhat'].squeeze()
        # form RL target
        if self.TDupdate: # actor-critic loss
            delta = tr.Tensor(expD['reward'][:-1])+self.gamma*vhats[1:].squeeze()-vhats[:-1].squeeze()
            delta = tr.cat([delta,tr.Tensor([0])])
        else: # REINFORCE
            delta = tr.Tensor(returns) - vhats
        # form RL loss
        distr = expD['pi']
        actions = expD['action']
        los_pi = tr.mean(delta*distr.log_prob(actions))
        ent_pi = distr.entropy().mean() # mean over time
        los_val = tr.mean(tr.square(returns - vhats)) # MSE
        los = 0.1*los_val-los_pi
        # update step
        self.optiop.zero_grad()
        los.backward()
        self.optiop.step()
        return None 




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



if __name__ == "__main__":
    ## setup
    task = PWMTask(stim_set=[(0,1),(1,0)])
    actor = ActorCritic()
    ## forward single trial
    stateL,obsL=task.sample_trial(trlen=4)
    pi_distr,vhatA = actor.unroll_trial(obsL)
    ## perform single update
    env = Env(actor,task)
    data = env.run_pwm_trial(delay=2,update=True)
    