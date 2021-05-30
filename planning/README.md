# the frustrated rodent simulation

## phenomenon 
* rodents get full water alotment at the end of the day
* if they violate a trial, punishment is timeout, they can't start new trial. 
* if they find the task gets too difficult, they stop doing trials / violate 100% 
* _would rather wait until end of day to get their water instead of doing this ridiculous task_. 


## simulation
Could be simulated by simulating a full day on each epoch. 
* every epoch involves not one but a sequence of episodes. 
* et the end of the epoch, the agent gets alotted unclaimed reward, so that total reward is always same per epoch, but reward distribution over time differs.
* manipulate discount factor to simulate preference for rewards now vs just waiting until end of day.
* simulate timeout violation punishment by setting epoch length fixed, and allowing agent to do more trials if fewer violations.