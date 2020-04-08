#Note when gradient is decreasing than it is called gradient descent
# learning rate(gamma or alpha) The learning rate determines how big the step would be on each iteration.
 # The most commonly used rates are : 0.001, 0.003, 0.01, 0.03, 0.1, 0.3 range( .0001 to 10)
           
'''loss fun: Machines learn by means of a loss function. It’s a method of evaluating how well specific algorithm models the 
           given data. If predictions deviates too much from actual results, loss function would cough up a very large number.
           for example loss fun for house pricing models to (y) = (x-5)^2
           
           Note loss fun/objective fun/cost fun all are same with diff name
'''
#optimization eqn: x_ = x_ - learning rate * gradient(slope) of loss fun
               #note: x_ is for  updating weights
#gradient = 1st derivative of loss function (in other word dy/dx i.e dy/dx = m or 1st derivative of given loss fun)
#note1 whenever a function is applied on a hidden layrer, it is callled activation fun and when same or diff fun is applied on out
# layer than it is callled output/probalistic/regress fun 

#1 Forward propagation is just used to get the probable output (y^), real output y we already have (eg linear reg- x1 x2 x3(Input, y(output))

# In backward propagation weights get updated according to error, error is due to diff between observed output and actual output
#there are lots of example of error fun, common one like mean square root error(err fun) eg 1/2(y-y^)² = 1/2(y-(wa°+b))² 
# a° is input(like x), Z° is aggregate input (wa° or wx) 


import numpy as np
import matplotlib.pyplot as plt


x = np.arange(10)
print(x)




def loss_fun(x):
    return (x-5)**2

loss = loss_fun(x)

plt.style.use('seaborn')
plt.plot(loss)
#plt.show()                # as plt.show end the image there itself so dont use there as more thing as scatter plot is still needed to add


def gradient_dec():
    x_ = 0           # x_ is updated weights
    l_rate = 0.1     # learning_rate is nothing but to increase or decrease the step size for conversion
    
    print(x_)
   
    
    
    for i in range(25):               # that 25 is nothing but no of epoch to get loss to 0 as earliest
        gradient = 2*(x_-5)           # it is nothing but 1st derivative of loss(cost) function
         
        x_ = x_- l_rate * gradient    # i value for each iteration is changed in value of x_ (not in var x_)
        
        l = loss_fun(x_)              #here actually loss(l) fun start with 16 as x_ = 1 in 1st interation for x_ = 0
        print("x_: %.3f l: %.3f"  %(x_, l))
        
        plt.scatter(x_, l)
       
        
        
gradient_dec()
