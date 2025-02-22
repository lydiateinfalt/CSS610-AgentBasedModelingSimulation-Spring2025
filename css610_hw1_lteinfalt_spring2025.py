# -*- coding: utf-8 -*-
"""

CSS 610 Spring 2025 Homework 1 - Lydia Teinfalt 
(Spyder using Python 3.12 on HP laptop)
01/28/2025
"""

# import necessary libraries
import numpy as np
import matplotlib.pyplot as plt

"""Problem 1: Compute and display the sum of the first 1000 integers. Repeat 
for the first 10^5 integers"""

# declaring global variables
n = 1000
y = 10**5

# initialize sum to zero
sum1000 = 0
array1000 = [0] * n

for i in range(1, len(array1000)+1):
    array1000[i-1] += i
    sum1000 = sum1000 + i
print("Problem 1. Sum of first ", n, " integers = ", f"{sum1000:,}")

# initialize sum variable and array for 10^5 integers
sum10exp5 = 0
array10exp5 = [0] * y
for j in range(1, len(array10exp5)+1):
    array10exp5[j-1] += j
    sum10exp5 = sum10exp5 + j


print("Problem 1a. Sum of first ", y, " integers = ", f"{sum10exp5:,}")
print("----------------------------------------------------------------------------------------------")

"""Problem 2. (Simple statistics) Compute and display the average and 
(unbiased) standard deviation of the first 1000 integers. 
Repeat for the first 10^5 integers. Answers: 500.5 and
288.19 for the first part, 50,000.5 and 28,867.7 for the second part.
"""

# calculate mean by taking the sum of the array of integers and diving it by 
#the length of the array
avg1000 = sum1000/n
print("Problem 2. Average of first ", n, " integers = ", avg1000)

# initializing an array of size 1000
sd1000 = [0]*n

# initializing sum variable to 0
sd1000_sum = 0

# for loop ends before the max number in range
for i in range(1, len(array1000)+1):
    # index starts at 0
    sd1000[i-1] = (array1000[i-1]-avg1000)**2
    sd1000_sum = sd1000_sum + sd1000[i-1]
# unbiased sd calculating by sum of variance dividing by sample size n -1 and 
#taking a square root
z = (sd1000_sum/(n-1))**0.5
print("Problem 2. Standard deviation of first ",
      n, " integers = ", f"{z:,.2f}")

# Repeat same process but for first 100,000 integers
avg10exp5 = sum10exp5/y
print("Problem 2a. Average of first ", y, " integers = ", f"{avg10exp5:,.2f}")

# initializing an array of size 1000
var10exp5 = [0]*y

# initializing sum variable to 0
sd10exp5_sum = 0

# for loop ends before the max number in range
for k in range(1, len(array10exp5)+1):
    # index starts at 0
    var10exp5[k-1] = (array10exp5[k-1]-avg10exp5)**2
    sd10exp5_sum = sd10exp5_sum + var10exp5[k-1]
# unbiased sd calculating by sum of variance dividing by sample size n -1 and 
# taking a square root
z1 = (sd10exp5_sum/(y-1))**0.5
print("Problem 2a. Standard deviation of first ",
      y, " integers = ", f"{z1:,.2f}")
print("----------------------------------------------------------------------------------------------")

"""Problem 3. (Arrays) Instantiate an array of size 1000, fill it with 
the first 1000 integers, and then compute the average value and standard 
deviation by summation (as in #2 above, so the answers are the same)."""

# Instead of using for loop to find answers use numpy package
# Numpy has a built in arange method for creating sequence of integers
a = np.arange(1, n+1)

# numpy has buit-in sum method for summing the array
print("Problem 3. Sum of first 1000 integers = ", f"{np.sum(a):,.2f}")

# numpy has a built in arange method for creating sequence of integers
a1 = np.arange(1, y+1)
print("Sum of first 10^5 integers = ", f"{np.sum(a1):,.2f}")
print("----------------------------------------------------------------------------------------------")

# numpy has a built in mean that can compute the average of the array
# from first problem with first 1000 integers
avg = np.mean(a)
print("Average of first 1000 integers = ", f"{avg:,.2f}")

# numpy has a built in standard deviation function.
# to get the unbiased std, we need to send ddof=1 to the function
# which we execute on array "a" from first problem set
std = np.std(a, ddof=1)
print("Standard deviation of first 1000 integers = ", f"{std:,.2f}")
############### Repeating same stats for first 10^5 integers ##############
avg1 = np.mean(a1)
print("Average of first 10^5 integers = ", f"{avg1:,.2f}")
std1 = np.std(a1, ddof=1)
print("Standard deviation of first 10^5 integers = ", f"{std1:,.2f}")
print("----------------------------------------------------------------------------------------------")

"""Problem 4. (Distributed algorithm) With the array from #3 above (or some 
comparable data structure), perform the following computation: select two 
elements of the array at random, average them, and assign them this average; 
repeat this some large number of times and occasionally look at/print out the 
average and standard deviation of the entire array and describe how they 
change over time and why. If instead of computing the average of the entire 
array you simply selected an element at random as achange as the number of 
iterations increases? How do your results change if you select more than 
two elements to average?

Responses to Problem 5: For a sequence of numbers from 1...1000 describe how 
the average and standard deviation change over time and why.
1. By selecting any two values and averaging them and replacing them, the 
average value declines over the number of iterations until it stabilizes at 
around 493.0. Once the average value is ~ 493.0, the value does not change 
until end of iterations. This is the "center" of an 1000-dimension array is 
around 493.0 so the averaging process pulls the mean towards the center.
2. The standard deviation also declines but more rapidly towards 0 over time. 
The process of replacing an average of two values in the 1000 array creates 
uniformity and reduces variability to 0.
3. Using the method of selecting an element at random and treating it as a 
typical value, it also converges towards the average value in less than 25 
iterations. So increasing beyond 25 values does not impact the convergence 
process towards the average value.
4.  When three elements are used for the averaging process of the array, the 
convergence towards average value of about 493.0 happens quicker and uniformity 
is reached more rapidly than in the case of two elements.

"""

# Initialize variables
num = 1000
iterations = 200
a = a = np.arange(1, num+1)
t = [0]*iterations
sd = [0]*iterations
x = [0]*iterations

# Iterate the process of randomly selecting integers from the array and 
# averaging them 100 times
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
    # print(new_mean)
    new_sd = np.std(a, ddof=1)
    # print(new_sd)
    t[v] = new_mean
    sd[v] = new_sd

    # selected an element in this case a[r] as typical value
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

"""Problem 5. Make multiple realizations of the process in step 4, describe 
the differences from run to run and discuss the  origin of these differences.

Responses to Problem 5: Each run is slightly different due to the randomness 
of the selection of elements to do the averaging process. This is the 
effect of stochasity seen with instantiating the process multiple times and 
selecting randomly elements to include in the averaging process.
"""


def sequence_generator(n):

    # Initialize variables
    num = 1000
    iterations = 200
    a = a = np.arange(1, num+1)
    t = [0]*iterations
    sd = [0]*iterations
    x = [0]*iterations

    # Iterate the process based on iterations parameter of randomly selecting 
    # integers from the array and averaging them
    for v in range(iterations):
        for i in range(1, num+1):

            # randomly selects three integers from the array and averages them
            r = np.random.randint(num)
            s = np.random.randint(num)
            p = np.random.randint(num)

            new_avg = np.mean([a[r], a[s], a[p]])

            a[r] = new_avg
            a[s] = new_avg
            a[p] = new_avg

        new_mean = np.mean(a)
        # print(new_mean)
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


# number of runs parameter runs
runs = 50
# set the seed to 888
np.random.seed(888)
# max number in integer sequence (q)
q = 1000
for r in range(1, runs):
    sequence_generator(q)

"""Repository link: https://github.com/lydiateinfalt/CSS610-AgentBasedModelingSimulation-Spring2025/blob/main/css610_hw1_lteinfalt_spring2025.py

"""
