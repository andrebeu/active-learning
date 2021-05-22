# toy simulation 
- add task description
- add simulation description
- add goal state

# working notes

## how to handle rnn unroll ? (05/22)
- unsure how to handle RNN unroll
- online unrolls are more flexible
  - flexibility is required when actions interact with environment state
- however, more efficient to unroll rnn on all stimuli within trial, then apply output layers in parallel
- support for online is underway but incomplete.
- will start with rnn_unroll