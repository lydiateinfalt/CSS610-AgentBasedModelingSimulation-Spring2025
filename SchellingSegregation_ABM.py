# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 10:30:24 2025

@author: Lydia Teinfalt CSS 610 Agent-Based Model
Schelling Segregation Model
"""

import numpy as np
import random
from datetime import datetime

#Agent class that has the attributes: id, location, team, happy based on global threshold
class Agent:
    def __init__(self, id, location, team):
        self.id = id
        self.location = location
        self.team = team
        self.happy = False

    def update_happiness(self, neighbors, threshold):
        like_neighbors = sum(1 for neighbor in neighbors if neighbor and neighbor.team == self.team)
        all_neighbors = len(neighbors)
        #print(f"Agent {self.id} has {like_neighbors} like neighbors out of {all_neighbors} total neighbors.")
        happy_quotient = (like_neighbors/all_neighbors) >= threshold
        #print(f"Agent {self.id} is happy: {happy_quotient}")
        if all_neighbors == 0:
            self.happy = False
        else:
          self.happy = happy_quotient

    def __repr__(self):
        return f"Agent(id={self.id}, location={self.location}, team={self.team}, happy={self.happy})"

#Location class that holds one agent
class Location:
    def __init__(self):
        self.agent = None

    def is_empty(self):
        return self.agent is None

class TwoDLocation:
    def __init__(self, grid_size):
        self.grid = np.full((grid_size, grid_size), None)

    def place_agent(self, agent, x, y):
        self.grid[x, y] = agent
        agent.location = (x, y)
    
    def get_neighbors(self, x, y):
        neighbors = []
        n, m = self.grid.shape

        # Define the relative positions of the neighbors
        neighbor_positions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for dx, dy in neighbor_positions:
            nx, ny = (x + dx) % n, (y + dy) % m
            neighbors.append(self.grid[nx, ny])

        return neighbors

    def move_agent(self, agent, new_x, new_y):
        old_x, old_y = agent.location
        self.grid[old_x, old_y] = None
        self.place_agent(agent, new_x, new_y)                

    def __repr__(self):
        grid_repr = ""
        for row in self.grid:
            row_repr = ""
            for cell in row:
                if cell is None:
                    row_repr += ". "
                else:
                    row_repr += f"{cell.team} "
            grid_repr += row_repr + "\n"
        return grid_repr

class Population:
    def __init__(self, grid_size, num_agents, teams, threshold):
        self.grid_size = grid_size
        self.locations = TwoDLocation(grid_size)
        self.agents = []
        self.teams = teams
        self.threshold = threshold
        self.initialize_agents(num_agents)

    def initialize_agents(self, num_agents):
        for i in range(num_agents):
            location = self.get_random_empty_location()
            team = random.choice(self.teams)
            agent = Agent(i, location, team)
            self.agents.append(agent)
            self.locations.place_agent(agent, location[0], location[1])

    def get_random_empty_location(self):
        while True:
            x = random.randint(0, self.grid_size-1)
            y = random.randint(0, self.grid_size-1)
            if self.locations.grid[x][y] is None:
                return (x, y)

    def update_happiness(self):
        for agent in self.agents:
            neighbors = self.locations.get_neighbors(*agent.location)
            agent.update_happiness(neighbors, self.threshold)

    def move_agents(self):
        for agent in self.agents:
            if not agent.happy:
                #moves agents to a location they would be happy with
                #new_location = self.find_happy_location(agent)
                #moves agent to a randomly empty location
                new_location = self.get_random_empty_location()
                if new_location:
                    self.locations.move_agent(agent, new_location[0], new_location[1])
  
    def find_happy_location(self, agent):
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if self.locations.grid[x, y] is None:
                    neighbors = self.locations.get_neighbors(x, y)
                    like_neighbors = sum(1 for neighbor in neighbors if neighbor and neighbor.team == agent.team)
                    if like_neighbors >= self.threshold:
                        return (x, y)
        return None

    def calculate_average_like_neighbors(self):
        total_like_neighbors = 0
        total_neighbors = 0
        for agent in self.agents:
            neighbors = self.locations.get_neighbors(*agent.location)
            like_neighbors = sum(1 for neighbor in neighbors if neighbor and neighbor.team == agent.team)
            total_like_neighbors += like_neighbors
            total_neighbors += len(neighbors)
        if total_neighbors == 0:
            return 0
        return total_like_neighbors / total_neighbors

    def __repr__(self):
        return f"Population(grid_size={self.grid_size}, agents={self.agents})"

#Run_sim method for running the simulation multiple times
#grid_size the number of rows x number of columns
#teams based on a list in the caller function
#threshold is the lower preference ratio of agent's for similar neighbor
#neighborhood is von Neummann
#num_runs tells how many times to run the simulation
#max_moves is the maximum number of times agents can move -- executions were taking too long
def run_sim(grid_size, num_agents, teams, threshold, num_runs, max_moves):
    stats = []
    for run in range(num_runs):
        startTime = datetime.now()
        population = Population(grid_size, num_agents, teams, threshold)
        #print(f"Initial grid for run {run + 1}:")
        #print(population.locations)
        all_happy = False
        move_count = 0
        while not all_happy and move_count < max_moves:
            population.update_happiness()
            all_happy = all(agent.happy for agent in population.agents)
            if not all_happy:
                population.move_agents()
                move_count += 1
        #print(f"Final grid for run {run + 1}:")
        #print(population.locations)
        stats.append(move_count)
        if move_count >= max_moves:
            print(f"Run {run + 1} reached the maximum move limit of {max_moves}.")
        average_like_neighbors = population.calculate_average_like_neighbors()
        print(f"Average like neighbors: {average_like_neighbors:.2f}")
        endTime = datetime.now()
        print ("Model execution time (HH:MM:SS) is: " + str(endTime-startTime))
    return population, stats


# Instantiate the Schelling Segregation Model simulation run with the following initial parameters
teams = ["X", "O"]
grid_size = 10 #number of rows by number of columns landscape
agents_per_matrix = round(0.80 * grid_size * grid_size) #ratio of matrix not empty
threshold = 0.625 # Ratio of like neighbors for agents to be happy
simulation_runs = 10  # Number of times to run the simulation

# Run the model with where there is a parameterized max number of times agents 
# can move to be happy
# returns population for the run
population_run, simulation_results = run_sim(grid_size, agents_per_matrix, teams, threshold, simulation_runs, max_moves=10000)

# Display the number of moves for each run
print(f"Simulation results over {simulation_runs} runs:")
for i, moves in enumerate(simulation_results):
    print(f"Run {i + 1}: Total moves = {moves}")