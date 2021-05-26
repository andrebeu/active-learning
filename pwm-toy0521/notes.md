# notes / plans
Question: does curriculum delay help learning?
Target: find conditions: pretraining on partial task helps


## experiment: growing delay

How does delay time between first and second stimulus affect training?
Experiment setup:
- two one-hot stimuli
- reward +1 hold, +1 correct action, 0 otherwise
- vary number of timesteps between first and second stimuli

Result: Shorter delays train faster on selecting rewarded action
- Not surprising, shorter term credit assignments are easier.

Result: Longer delays train faster on holding
- Consequence of training protocol: One backprop per trial; since longer delays have more holding timesteps, each trial has more rewarded hold timesteps.
  - future experiments should try to control for this

Exp: does pretraining on shorter delay help acquisition of longer delay?


## experiment: stimulus similarity

goal: simulate easy vs difficult trial
 * show acc rises faster for easy when trained on both concurrently

## reward structure note

* should I reward hold or punish hold violation?
* noticed that training hold on longer delays allows model to learn hold _faster_ than shorter delays, because more holding experience per episode. 
  * hold reward could scale with number of steps 
  * might need to be careful with equating batch sizes when comparing different delay conditions.


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