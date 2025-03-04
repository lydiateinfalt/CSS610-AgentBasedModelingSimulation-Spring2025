# -*- coding: utf-8 -*-
"""
Course: CSS 610 Spring 2025
Author: Lydia Teinfalt
Created on Sat Mar  1 15:55:45 2025
Edgeworth Box

@author: petit
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import random

# Set a random seed for reproducibility
np.random.seed(14248)
K = 25 #Amount of goods 1
L = 15 #Amount of goods 2

# Create Consumer Class
# Give them Cobb-Douglas preferences
# Give them random initial endowments
class Consumer:
    def __init__(self, id, alpha, beta, endowment1, endowment2):
        self.id = id
        self.alpha = alpha
        self.beta = beta
        self.endowment1 = endowment1
        self.endowment2 = endowment2

    # Define the utility functions for the consumer
    def utility(self, x, y):
        return (x**self.alpha) * (y**self.beta)

    # Define the MRS for the consumer
    def MRS(self, x, y):
        return (self.alpha / self.beta) * (y / x)

# Create Population Class
# Generate N Agents
# Randomly select two agents in N classes to exchange
class Population:
    def __init__(self, N):
        list = [0.25,0.33, 0.5]
        alpha = random.choice(list)
        beta = random.choice(list)
        self.consumers = [Consumer(i, alpha, beta, np.random.randint(1, K), np.random.randint(1, L)) for i in range(N)]
        self.pareto = False
        self.trade_history = []

    # Function to check if the population is Pareto optimal
    def pareto_optimal(self):

        consumer_indices = list(range(len(self.consumers)))
        np.random.shuffle(consumer_indices)

        for i in range(len(consumer_indices)):
            for j in range(i + 1, len(consumer_indices)):
                if self.trade_possible(self.consumers[i], self.consumers[j]):
                    return False
        return True

    # Function to check if trade is possible between two consumers
    def trade_possible(self, consumer1, consumer2):
        MRS1 = consumer1.MRS(consumer1.endowment1, consumer1.endowment2)
        MRS2 = consumer2.MRS(consumer2.endowment1, consumer2.endowment2)
        if MRS1 != MRS2:
            return True
        elif MRS1 == MRS2:
            return False
        elif math.isclose(MRS1, MRS2):
            return False
        else:
            return True

    # Function to initiate trade between two consumers
    def trade(self):
        consumer_indices = list(range(len(self.consumers)))
        np.random.shuffle(consumer_indices)

        for i in range(len(consumer_indices)):
            for j in range(i + 1, len(consumer_indices)):
                consumer1 = self.consumers[consumer_indices[i]]
                consumer2 = self.consumers[consumer_indices[j]]
                if self.trade_possible(consumer1, consumer2):
                    #print(f"Trade possible between Consumer {consumer1.id} and Consumer {consumer2.id}")
                    self.execute_trade(consumer1, consumer2)
                    # Check for Pareto optimality after each trade
                    if self.pareto_optimal():
                        print("Pareto Optimality achieved after trade")
                        return
                    return

    # Function to execute trade between two consumers by picking a value from the contract curve
    def execute_trade(self, consumer1, consumer2):
        # Pick a random value of x within the range of their endowments
        x_new = random.uniform(consumer1.endowment1, consumer2.endowment1)
        
        #total endowments of consumers 1 and 2
        x_total = consumer1.endowment1 + consumer2.endowment1
        y_total = consumer1.endowment2 + consumer2.endowment2


        # Calculate the corresponding y value on the contract curve
        y_new = (consumer1.beta*(1-consumer1.alpha)*x_new*x_total)/(consumer1.alpha*x_total*(1-consumer1.beta)-(x_new*(consumer1.alpha - consumer1.beta)))
        
        # Record the trade history before updating endowments
        self.trade_history.append(((consumer1.endowment1, consumer1.endowment2), (consumer2.endowment1, consumer2.endowment2), (x_new, y_new)))

        #assign the new randomly selected endowments to first consumer
        consumer1.endowment1 = round(x_new, 2)
        consumer1.endowment2 = round(y_new, 2)

        #Second consumer's new endowments based on difference between max dimensions of the Edgeworth box and consumer1's endowments
        consumer2.endowment1 = round((K - consumer1.endowment1), 2)
        consumer2.endowment2 = round((L - consumer1.endowment2), 2)
        
        # Print the trade details for debugging
        #print(f"Executing trade: Consumer {consumer1.id} gets ({consumer1.endowment1}, {consumer1.endowment2}), Consumer {consumer2.id} gets ({consumer2.endowment1},{consumer2.endowment2})")


# Create a simulation class that instantiates a population of N consumers
# Selects two agents in the population and initiate trades
# Stops simulation once all agents have traded and they are Pareto optimal
def simulation(N, iterations):
    population = Population(N)
    for i in range(iterations):
        if population.pareto_optimal():
            print(f"Pareto Optimality reached at iteration {i}")
            population.pareto = True
            break
        else:
            population.trade()
    return population

# Function to visualize trades
def visualize_trades(population):
    plt.figure(figsize=(10, 8))

    for trade in population.trade_history:
        initial_A, initial_B, new_A = trade

        plt.plot([initial_A[0], new_A[0]], [initial_A[1], new_A[1]], 'ro-')
        plt.plot([initial_B[0], new_A[0]], [initial_B[1], new_A[1]], 'bo-')

    plt.xlabel('Good X')
    plt.ylabel('Good Y')
    plt.title('Consumer Trades Visualization')
    plt.grid(True)
    plt.show()

# Run simulation with different parameters and visualize trades
num_agents = [20]
num_iterations = [100000]
run_count = 1
total_runs = 100
trade_count = []
pareto_count = []
for p in range(total_runs):
    for i in num_agents:
        for j in num_iterations:
            print("*************************************************************")
            print(f"Run #{run_count}")
            print(f"Number of agents {i} and max number of iterations {j}")
            pop = simulation (i,j)
            trade_count.append(len(pop.trade_history))
            print("Number of trades = ", len(pop.trade_history))
            pareto_count.append(pop.pareto)
            #visualize_trades(pop)
            run_count = run_count + 1
    print(f"Average Number of Trades {sum(trade_count)/len(trade_count)}")
    count = pareto_count.count(True)
    print(f"Number of times pareto optimal solution reached {count}")

    


