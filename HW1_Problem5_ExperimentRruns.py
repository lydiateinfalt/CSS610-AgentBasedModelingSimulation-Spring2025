# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 12:53:04 2025

@author: Lydia Teinfalt
CSS 610 ABM Homework #1
"""
#import necessary libraries
import numpy as np
import matplotlib.pyplot as plt

#create a function from problem #4 that can be called on for 
#number of runs
def sequence_generator(n):

     # Initialize variables
    num = 1000
    iterations = 200
    a = a = np.arange(1, num+1)
    t = [0]*iterations
    sd = [0]*iterations
    x = [0]*iterations
    
    # Iterate the process of randomly selecting integers from the array and averaging them 100 times
    for v in range(iterations):
        for i in range(1, num+1):
            r = np.random.randint(num)
            s = np.random.randint(num)
            p = np.random.randint(num)
    
            new_avg = np.mean([a[r], a[s], a[p]])
    
            a[r] = new_avg
            a[s] = new_avg
            a[p] = new_avg
            
        new_mean = np.mean(a)
        new_sd = np.std(a, ddof=1)
        t[v] = new_mean
        sd[v] = new_sd
        x[v] = (a[r] - new_mean) / new_mean
    
    # Plot the results
    plt.figure(figsize=(12, 6))
    
    # average
    plt.subplot(3, 1, 1)
    plt.plot(t, label='Mean')
    plt.title('Changes to Average Value over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Average')
    plt.legend()
    # standard deviation
    plt.subplot(3, 1, 2)
    plt.plot(sd, label='Standard Deviation', color='red')
    plt.title('Changes to Standard Deviation over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Standard Deviation')
    plt.legend()
    # normalized value
    plt.subplot(3, 1, 3)
    plt.plot(x, label='Normalized Value', color='green')
    plt.title('Changes to Normalized Value over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Normalized Value')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Problem 5 Make multiple realizations of the process in step 4, describe the differences
# from run to run and discuss the origin of these differences.
runs = 100
#set the seed to 888
np.random.seed(888)
#max number in integer sequence
q = 1000
for r in range (1,runs):
  sequence_generator(q)
