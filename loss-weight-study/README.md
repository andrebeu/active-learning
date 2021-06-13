# loss weight study
* how does policy loss and value loss balance each other out?
* havent found a principled way to choose critic vs actor loss weights. 

## results pwm0
* unlike other applications, where I used a critic loss weight less than one, it looks like better performance is attained with having the critic (value) loss weight be an order of magnitude greater than the policy loss. 

## note: pwm0
* pwm0 is the trivial version of the task, where response is entirely given by second stimulus

## note: critical bug 05/12
* when unrolling the environment, I was previously collecting data by using `list.extend()`. when extending a list with pytorch tensors, the tensor object lost track of gradient information. 
