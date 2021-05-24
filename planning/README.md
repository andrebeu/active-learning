# notes and plans

## the frustrated rodent simulation
* rodents get full water alotment at the end of the day
* if they find the task gets too difficult, they stop doing trials / violate 100% of the trials because they would rather wait until the end of the day to get their water
* if they violate a trial, punishment is timeout, they can't start new trial. 

Could be simulated by simulating a full day on each epoch. 
* discount factor might allow us to simulate rodents preference for rewards now vs just waiting until end of day.
* timeout violation punishment could be simulated by setting epoch length fixed, and allowing agent to do more trials if fewer violations.