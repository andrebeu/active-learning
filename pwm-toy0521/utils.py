import numpy as np
import torch as tr
from collections import namedtuple
from torch.distributions.categorical import Categorical


""" next steps
- left a bunch of asserts that need to be verified
- fix agent.update, being called in env.run_pwm_trial
"""


# variable delay trial
class Env():
    def __init__(self,actor,task):
        self.actor = actor
        self.task = task
        return None

    def reward_fn(self,stateL,obsL,actionL):
        """ reward_hold_and_lastaction
        state number indicates rewarded action
        reward hold everywhere except 
         after second stimulus
        """
        trlen = len(stateL)
        reward = -np.ones(trlen)
        reward_hold = (actionL[:-1] == 0)
        reward_action = int(stateL[-1] == actionL[-1])
        reward = np.concatenate([reward_hold,reward_action])
        print('s',stateL,'a',actionL,'r',reward)
        assert False
        return rt


    def run_pwm_trial(self,delay,update=True):
        """ 
        sample trial
        perform model update

        """
        stateL,obsL = self.task.sample_trial(delay+2)
        pi_distr,vhatA = self.actor.unroll_trial(stateL)
        actionL = pi_distr.sample()
        assert len(actionL)==len(obsL)
        rewardL = self.reward_fn(stateL,actionL)
        if update:
            self.actor.update()
        trial_data = {
            'states':stateL,
            'obss':obsL,
            'actions':actionL,
            'rewards':rewardL,
        }
        return trial_data
    
    



class PWMTask():
    
    def __init__(self,stim_set,delay):
        """ action space {0:hold,1:left,2:right}
        """
        self.stim_set = stim_set
        self.delay = delay
        self.action_set = [0,1,2]
        return None

    def sample_trial(self,trtype,trlen):
        stateL,obsL = self.sample_state_and_obs(trlen)
        actionL = unroll_agent(stateL)
        return stateL,obsL,rewardL

    def sample_trial(self,trlen):
        """ returns two arrays of `trlen`
        - action_t = actors(obs)
        - reward_t = reward_fn(state,action)
        """
        delay = trlen-2
        assert delay>0 # 2 stim
        SAi,SBi = self.stim_set[np.random.choice(len(self.stim_set))]
        # embed stim
        SA,SB = SAi,SBi 
        obsL = np.zeros(trlen) # two stim
        obsL[0] = SA
        obsL[-1] = SB
        # instructs rewarded action 
        stateL = np.zeros(trlen)
        stateL[-1] = 1+int(SA>SB) 
        return stateL,obsL


# class Agent(tr.nn.Module):
# class DQN(Agent):
# class AC(Agent):

class ActorCritic(tr.nn.Module):
    """ implementation NOTEs: 

    Two modes:
     rnn_step can be used online
      RNN state held as instance variable
     while rnn_unroll is more efficient
      returns states for every obs in sequence

    TODO: 
     - restructure learning target setup to allow 
     TD lambda continuum between onestep TD and MC 

    """
  
    def __init__(self,indim=1,nactions=3,stsize=15,gamma=0.80,learnrate=0.005,TDupdate=False):
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

    def act(self,obs):
        """ 
        pi_act [batch,nactions] is output of policy head
        which gets passed through a softmax 
        and from which an action is sampled
        """
        hstate = self.rnn_step(obs)
        pi_act = self.rnn2pi(hstate_t)
        pism = pi_act.softmax(-1)
        pidistr = Categorical(pism)
        action = pidistr.sample()
        return action

    def rnn_step(self,obs):
        """ 
        updates rnn state instance variable
         flexible: used for online 
        """
        obs = tr.Tensor(obs).unsqueeze(0)
        self.h_t,self.c_t = self.rnn(obs,(self.h_t,self.c_t))
        return self.h_t

    def rnn_unroll(self,obsL):
        """ 
        
        """

        return None

    def unroll_trial(self,obsL):
        """
        given sequence of stimuli, unrolls rnn
         returns actions and hidden states
        efficient: forward layers applied in parallel
         """
        # propo rnn and forward pass head layers
        rnn_out,h_n = self.rnn(obsL)
        assert len(h_n) == 1
        assert len(rnn_out) == len(obsL)
        vhatA = self.rnn2val(rnn_out)
        piact = self.rnn2pi(rnn_out)
        pism = piact.softmax(-1) 
        # probability of each action each timestep
        # batch dim might be included
        assert pism.shape == (len(obsL),self.nactions)
        """
        distribution constructor 
         takes probabilities of each category
        returns distribution object
        distr.log_prob required 
        """
        pi_distr = Categorical(pism)
        assert False
        return pi_distr,vhatA

    def update_trial(self,obsL,rewardL):
        """ performs update on single trial """
        return None

    def update(self,expD):
        """ 
        supported REINFORCE and A2C updates
         batch update given expD trajectory:
        expects expD = {'reward':[tsteps],'state':[tsteps],...}
        compute output layers within (i.e. expects output of rnn unroll)
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
        actionL = pi_distr.sample()
        assert len(actionL) == len(obsL)
        logprob_actions = pi_distr.log_prob(actionL)

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

    ### 



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
            exp = Exp('tstep',obs[0],action,r_t,obs[0]+1,h_t)
            EpBuff.append(exp)
        return EpBuff

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


# EXP and unpackexpL might not be needed
""" 
i dont forsee using exp buffer
output of trial unrolls are DoL already
"""
Exp = namedtuple('Experience',[
    'state','obs','action','reward',
    'obs_tp1','rnn_state'
    ], defaults=[None]*2)


def unpack_expL(expLoD):
  """ 
  given list of experience (namedtups)
      expLoD [{t,s,a,r,sp}_t]
  return dict of np.arrs 
      exp {s:[],a:[],r:[],sp:[]}
  """
  expDoL = Exp(*zip(*expLoD))._asdict()
  return {k:np.array(v) for k,v in expDoL.items()}


def play_trial_unroll(self,agentpi,task):
    """ deprecated
    action <- agentpi(state)
    task methods: sample_trial, reward_fn 
    """
    trial_st,trial_obs = task.sample_trial()
    expL = []
    final_state = False
    for tstep in range(len(trial_st)):
        if tstep == len(trial_st)-1:
            final_state = True
        ## play
        st = trial_st[tstep]
        ot = trial_obs[tstep]
        at = agentpi(ot)
        rt = task.reward_fn(st,at)
        # final state
        if final_state:
            otp = -1
        else:
            otp = trial_st[tstep+1]
        # record
        expt = Exp(st,ot,at,rt,otp)
        expL.append(expt)
    return expL