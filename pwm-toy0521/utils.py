import numpy as np
import torch as tr
from collections import namedtuple
from torch.distributions.categorical import Categorical


Exp = namedtuple('Experience',[
    'tstep','state','action','reward','state_tp1','rnn_state'
    ],defaults=[None]*6)

class PWMTask():
    """ 
    WM tasks are POMDP environments
    transitions do not depend on action
    """
    
    def __init__(self,stim_set):
        self.stim_set = stim_set
        self.action_set = [0,1,2]
        self.randpi_fn = lambda x: np.random.choice(len(self.action_set))
        return None

    def reset_trial(self):
        """ initialize vars between trials"""
        self.tstep = 0
        self.state = 0
        return None

    def reward_fn(self,st,at):
        rt = 0
        return rt

    def step(self):
        None

    def play_trial(self,pifn=None):
        """ 
        distinguishing state from observation becasue POMDP
            R/L response depends on comparison between 
            first and last obs
        """
        self.reset_trial()
        if type(pifn)==type(None):
            pifn = self.randpi_fn
        # sample stimuli for trial
        SAi,SBi = self.stim_set[np.random.choice(len(self.stim_set))]
        SA,SB = SAi,SBi # int coding
        delay = 2
        trial_obs = np.zeros(delay+2) # two stim
        trial_obs[0] = SA
        trial_obs[-1] = SB
        """ trial_st
        instructs rewarded action 
        0 is hold, 1 is 1, 2 is R
        reward hold everywhere except 
        after second stimulus
        """
        trial_st = np.zeros(delay+2)
        trial_st[-1] = 1+int(SA>SB) 
        expL = []
        for tstep,(st,ot) in enumerate(zip(trial_st,trial_obs)):
            at = pifn(ot)
            rt = self.reward_fn(st,at)
            if tstep == len(trial_st)-1:
                stp = -1
            else:
                stp = trial_st[tstep+1]
            expt = Exp(tstep,st,at,rt,stp)
            expL.append(expt)
            print(expt)
        return expL
    
    



class ActorCritic(tr.nn.Module):
  
    def __init__(self,indim=4,nactions=2,stsize=18,gamma=0.80,learnrate=0.005,TDupdate=False):
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
        self.rnn = tr.nn.LSTMCell(self.indim,self.stsize,bias=True)
        self.rnn_st0 = tr.nn.Parameter(tr.rand(2,1,self.stsize),requires_grad=True)
        self.rnn2val = tr.nn.Linear(self.stsize,1,bias=True)
        self.rnn2pi = tr.nn.Linear(self.stsize,self.nactions,bias=True)
        # optimization
        self.optiop = tr.optim.RMSprop(self.parameters(), 
          lr=self.learnrate
        )
        return None

    def unroll_ep(self,task):
        """ actor logic 
        """
        finalst = False
        task.reset()
        self.h_t,self.c_t = self.rnn_st0
        EpBuff = []
        action = np.random.binomial(1,0.5)
        while not finalst:
            obs,r_t,finalst = task.step(action)
            h_t = self.rnn_step(obs)
            vh_t = self.rnn2val(h_t)
            pih_t = self.rnn2pi(h_t) 
            action = self.act(pih_t)
            exp = Experience('tstep',obs[0],action,r_t,obs[0]+1,h_t)
            EpBuff.append(exp)
        return EpBuff

    def rnn_step(self,obs):
        obs = tr.Tensor(obs).unsqueeze(0)
        self.h_t,self.c_t = self.rnn(obs,(self.h_t,self.c_t))
        return self.h_t

    def act(self,pi_out):
        """ pi_out [batch,nactions] is output of policy head
        """
        pism = pi_out.softmax(-1)
        pidistr = Categorical(pism)
        actions = pidistr.sample()
        return actions

    def eval(self,expD):
        """ """
        data = {}
        ## entropy
        vhat,pact = self.forward(expD['state'])
        pra = pact.softmax(-1)
        entropy = -1 * tr.sum(pra*pra.log2(),-1).mean()
        data['entropy'] = entropy.detach().numpy()
        ## value
        returns = compute_returns(expD['reward']) 
        data['delta'] = np.mean(returns - vhat.detach().numpy())
        return data

    def update(self,expD):
        """ REINFORCE and A2C updates
        given expD trajectory:
        expD = {'reward':[tsteps],'state':[tsteps],...}
        """
        # unpack experience
        rnn_states = tr.cat([*expD['rnn_state']])
        vhat = self.rnn2val(rnn_states)
        pact = self.rnn2pi(rnn_states)
        actions = tr.Tensor(expD['action'])
        reward = expD['reward']
        returns = compute_returns(expD['reward'],gamma=self.gamma) 
        # form RL target
        if self.TDupdate: # actor-critic loss
            delta = tr.Tensor(expD['reward'][:-1])+self.gamma*vhat[1:].squeeze()-vhat[:-1].squeeze()
            delta = tr.cat([delta,tr.Tensor([0])])
        else: # REINFORCE
            delta = tr.Tensor(returns) - vhat.squeeze()
        # form RL loss
        pi = pact.softmax(-1)
        distr = Categorical(pi)
        los_pi = tr.mean(delta*distr.log_prob(actions))
        ent_pi = tr.mean(tr.sum(pi*tr.log(pi),1))
        los_val = tr.square(tr.Tensor(returns) - vhat.squeeze()).mean()
        los = 0.1*los_val-los_pi+0.1*ent_pi
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
  T = len(rewards) 
  returns = np.array([
      np.sum(np.array(
          rewards[t:])*np.array(
          [gamma**i for i in range(T-t)]
      )) for t in range(T)
  ])
  return returns

def unpack_expL(expLoD):
  """ 
  given list of experience (namedtups)
      expLoD [{t,s,a,r,sp}_t]
  return dict of np.arrs 
      exp {s:[],a:[],r:[],sp:[]}
  """
  expDoL = Exp(*zip(*expLoD))._asdict()
  return {k:np.array(v) for k,v in expDoL.items()}

