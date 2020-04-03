#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
    x_ = 0           # x_ is nothing but optimal value, intially it can be any random value
    l_rate = 0.1
    
    print(x_)
   
    
    
    for i in range(25):               # that 25 is nothing but no of epoch to get loss to 0 as earliest
        gradient = 2*(x_-5)
         
        x_ = x_- l_rate * gradient    # i value for each iteration is changed in value of x_ (not in var x_)
        
        l = loss_fun(x_)              #here actually loss(l) fun start with 16 as x_ = 1 in 1st interation for x_ = 0
        print("x_: %.3f l: %.3f"  %(x_, l))
        
        plt.scatter(x_, l)
       
        
        
gradient_dec()