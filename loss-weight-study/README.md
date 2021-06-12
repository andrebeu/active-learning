# the frustrated rodent simulation

## phenomenon 
* rodents get full water alotment at the end of the day
* if they violate a trial, punishment is timeout, they can't start new trial. 
* if they find the task gets too difficult, they stop doing trials / violate 100% 
* _would rather wait until end of day to get their water instead of doing this ridiculous task_. 

## simulation
Goal: explore conditions for frustrated rodent phenomenon 
* every epoch corresponds to a day, i.e. sequence of trials. 
* at the end of the epoch, the agent gets alotted unclaimed reward (pub), so that total reward is always same per epoch, but reward distribution over time differs.
* simulate timeout violation punishment by setting epoch length fixed, and allowing agent to do more trials if fewer violations.
* manipulate discount factor to simulate preference for rewards now vs just waiting until end of day.


# Experiments

## goals

* show delay curriculum mitigates frustrated rodent phenomenon
  * reward too sparse if training on full trlen from start
* effects of discount factor (and TD methods)? 

## questions
* discount factor or TD methods with shorter horizon?
* actor critic only or Qlearning as well?

## exp todo
* result: delay curriculum helps
  * verify trains with short TR len
  * verify no train with long TR len
  * delay curriculum experiment
* explore: discount factors
  * 
* explore: TD updates
  * 

# implementation 

## notes
* task and agent objects. `runepoch` function loops `task.sample_trail()`.
* agent response influences trial type (valid vs not-valid). not valid corresponds to intertrial interval, implemented as a zero-delay of length `trlen`.
* support different `trlen` for each epoch to allow delay-curriculum. epoch_len must be fixed. padding might be required when epoch len not divisible by trial len. although using epoch=len 60, trlen=2,3,4 require no padding.
* pub happens on final step (epoch_len+1)

## simplifications
* two stimuli, gaussian but maximally separated
* epoch length fixed at 60, ITI = trlen = 5 (delay 3)

## note on sparse reward
* currently failing to hold (during trial and ITI) makes the following trial not-valid. since not-valid trials have no reward opportunity, the envrionment reward is very sparse. might make learning difficult if agent gets stuck in violating every trial. might need to pretrain holding behavior to get learning off the ground. 

## note: critical bug 05/12
* when unrolling the environment, I was previously collecting data by using `list.extend()`. when extending a list with pytorch tensors, the tensor object lost track of gradient information. 
