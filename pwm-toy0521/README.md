# Question: does curriculum delay help learning?
# Target: find conditions: pretraining on partial task helps


## experiment: growing delay

Exp1: does PWM3 train faster than PWM6? 
Exp2: parametric delay 1-6
Exp3: pretraining on shorter delays help longer delay train time?

PWM task: 
  * two stimuli (SA,SB)
  * delay of 3 vs 6 steps

Agents:
  * DQN
  * AC (one step to MC)


## experiment: stimulus similarity

goal: simulate easy vs difficult trial
 * show acc rises faster for easy when trained on both concurrently

## note POMDP

* WM tasks are POMDP environments in which state(t+1) do not depend on action
* implication: unlike traditional RL, environment and agent can be disentangled 
* POMDP because: response depends on comparison between first and last obs

# working notes

## note on batching: currently 'online' mode, i.e. batchsize = 1 (05/24)
- should we use batched training, or should each epoch consist of a single trial?
  - online mode is more cognitively plausible / defensible. 
  - but online mode is less efficient
- since the first set of experiments are expected to be relatively low complexity, I am going with online. I left `assert batchsize=1` statements in places where changes would need to be made to accommodate batching.

## efficiency note: how to handle rnn unroll / env steps? (05/22)
- RNN unrolls (required in agent.update) with steps in trial (usually happens in environment method)
- online unrolls in environment method are more flexible
  - allows actions interact with environment state
  - but our case this is not needed (see note pomdp)
  - support for online is deprecated and incomplete.
- more efficient to unroll rnn on all stimuli within trial, then apply output layers in parallel
- the issue is these two options have implications for the implementations of environment `play_trial`  method, and the agent `update` method. 
- will start with rnn_unroll