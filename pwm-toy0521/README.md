- Question: does curriculum delay help learning?
- Target: find conditions: pretraining on partial task helps


# experiment: PWM 3vs6
Exp1: does PWM3 train faster than PWM6? parametric delay 1-6

PWM task: 
  * two stimuli (SA,SB)
  * delay of 3 vs 6 steps
Agents:
  * DQN
  * AC (one step to MC)
Goal:
  * 

# experiment: stimulus similarity

# working notes

## how to handle rnn unroll ? (05/22)
- unsure how to handle RNN unroll
- online unrolls are more flexible
  - flexibility is required when actions interact with environment state
- however, more efficient to unroll rnn on all stimuli within trial, then apply output layers in parallel
- support for online is underway but incomplete.
- the issue is these two options have implications for the implementations of environment `play_trial`  method, and the agent `update` method. 
- will start with rnn_unroll