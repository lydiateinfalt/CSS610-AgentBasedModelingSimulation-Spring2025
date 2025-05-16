import mesa
import numpy as np
import random
from mesa.space import SingleGrid
import matplotlib.pyplot as plt

from agents import ABP, ACFAgent

class MigrationModel(mesa.Model):
    """
    Model of migration and economic agglomeration with active Brownian particles.
    """
    def __init__(self, num_agents=80, width=10, height=10, hiring_rate=0.1, firing_rate=0.05, movement=True,
                 A=1.0, a1=0.1, a2=-0.01):  # Initialize parameters
        super().__init__()
        self.num_agents = num_agents
        self.grid = SingleGrid(width, height, torus=True)
        self.hiring_rate = hiring_rate
        self.firing_rate = firing_rate
        self.running = True
        self.movement = movement

        self.A = A # This is a constant parameter related to the base productivity when cooperative effects are negligible
        self.a1 = a1
        self.a2 = a2
        self.beta = 0.7 #Given beta value

        # Initialize agents - Half employed, half unemployed
        for i in range(self.num_agents):
            is_employed = i < num_agents / 2
            a = ABP(i, self, is_employed)
            # Random initial placement in an empty cell
            empty_cells = [(x, y) for x in range(width) for y in range(height) if self.grid.is_cell_empty((x, y))]
            if empty_cells:
                cell = random.choice(empty_cells)
                self.grid.place_agent(a, cell)
            else:
                print("No empty cells available!") # Handle the case where there are no empty cells

        # Initialize wage field (2D array).
        self.wage_field = np.zeros((width, height)) # Initialize with zeros

        # For collecting data
        self.datacollector = mesa.DataCollector(
            model_reporters={"Employed": lambda m: sum(1 for a in m.agents if a.is_employed),
                             "Unemployed": lambda m: sum(1 for a in m.agents if not a.is_employed),
                             "Employed_Share": lambda m: sum(1 for a in m.agents if a.is_employed) / m.num_agents,
                             "Unemployed_Share": lambda m: sum(1 for a in m.agents if not a.is_employed) / m.num_agents
                             },
            agent_reporters={"IsEmployed": "is_employed"}  # Example agent reporter
        )

    def step(self):
        """Advance the model by one step."""
        self.agents.shuffle_do("step")

        # Update wage field based on agent locations and employment.
        self.update_wage_field()
        self.datacollector.collect(self)

    def get_employed_ratio(self):
        """
        Calculates the ratio of employed agents to the total number of agents.

        Returns:
            float: The ratio of employed agents to the total number of agents.
                   Returns 0 if there are no agents to avoid errors.
        """
        try:
            return sum(1 for a in self.agents if a.is_employed) / self.num_agents
        except ZeroDivisionError:
            return 0  # Or you might want to return float('nan') for "Not a Number"

    def update_wage_field(self):
         """
         Based on Schweitzer's equation 
         Update the wage field using the provided formula, calculating luv over the Moore neighborhood.
         ωuv(t) = (Ā/2) * [1 + exp(a1 * luv(t) + a2 * (luv(t))^2)] * 0.5 * (luv(t))^(-0.5) +
                   (Ā/2) * exp(a1 * luv(t) + a2 * (luv(t))^2) * (a1 + 2 * a2 * luv(t)) * (luv(t))^0.5
         """
         min_wage = 0.1  # Example minimum wage value

         for u in range(self.grid.width):
             for v in range(self.grid.height):
                 # Get Moore neighborhood for the current cell
                 neighborhood = self.grid.get_neighborhood((u, v), moore=True, include_center=True)  # Include center!

                 total_employed_in_neighborhood = 0
                 neighborhood_size = len(neighborhood)

                 for cell in neighborhood:
                     total_employed_in_neighborhood += sum(1 for a in self.grid.get_cell_list_contents(cell) if a.is_employed)

                 # Calculate local labor density (luv) over the neighborhood
                 luv = total_employed_in_neighborhood / neighborhood_size if neighborhood_size > 0 else 0

                 # Apply the wage formula
                 if luv > 0:  # Avoid division by zero
                     term1 = (self.A / 2) * (1 + np.exp(self.a1 * luv + self.a2 * (luv**2))) * 0.5 * (luv**(-0.5))
                     term2 = (self.A / 2) * np.exp(self.a1 * luv + self.a2 * (luv**2)) * (self.a1 + 2 * self.a2 * luv) * (luv**0.5)
                     self.wage_field[u, v] = term1 + term2
                 else:
                     self.wage_field[u, v] = min_wage  


class ACFMigrationModel(mesa.Model):
    """
    Model of migration and economic agglomeration
    """
    def __init__(self, num_agents=80, width=10, height=10, hiring_rate=0.1, firing_rate=0.05, movement=True,
                 A=1.0, a1=0.1, a2=-0.01):  # Initialize parameters
        super().__init__()
        self.num_agents = num_agents
        self.grid = SingleGrid(width, height, torus=True)
        self.hiring_rate = hiring_rate
        self.firing_rate = firing_rate
        self.running = True
        self.movement = movement

        self.A = A # This is a constant parameter related to the base productivity when cooperative effects are negligibl
        self.a1 = a1
        self.a2 = a2
        self.beta = 0.7 #Given beta value

        # Initialize agents - Half employed, half unemployed
        for i in range(self.num_agents):
            is_employed = i < num_agents / 2

            agent_location = random.choice(list(self.grid.coord_iter()))
            a = ACFAgent(i, self, agent_location, random.randint(1, 5), random.uniform(15, 25), random.uniform(0.1, 0.5),
                         aspiration_factor=0.2, capability_factor=0.3, is_employed=is_employed)
            # Random initial placement in an empty cell
            empty_cells = [(x, y) for x in range(width) for y in range(height) if self.grid.is_cell_empty((x, y))]
            if empty_cells:
                cell = random.choice(empty_cells)
                self.grid.place_agent(a, cell)
            else:
                print("No empty cells available!") # Handle the case where there are no empty cells

        # Initialize wage field (2D array).
        self.wage_field = np.zeros((width, height)) # Initialize with zeros

        # For collecting data
        self.datacollector = mesa.DataCollector(
            model_reporters={"Employed": lambda m: sum(1 for a in m.agents if a.is_employed),
                             "Unemployed": lambda m: sum(1 for a in m.agents if not a.is_employed),
                             "Employed_Share": lambda m: sum(1 for a in m.agents if a.is_employed) / m.num_agents,
                             "Unemployed_Share": lambda m: sum(1 for a in m.agents if not a.is_employed) / m.num_agents
                             },
            agent_reporters={"IsEmployed": "is_employed"}  # Example agent reporter
        )

    def step(self):
        """Advance the model by one step."""
        self.agents.shuffle_do("step")

        # Update wage field based on agent locations and employment.
        self.update_wage_field()
        self.datacollector.collect(self)

    def get_employed_ratio(self):
        """
        Calculates the ratio of employed agents to the total number of agents.

        Returns:
            float: The ratio of employed agents to the total number of agents.
                   Returns 0 if there are no agents to avoid errors.
        """
        try:
            return sum(1 for a in self.agents if a.is_employed) / self.num_agents
        except ZeroDivisionError:
            return 0  # Or you might want to return float('nan') for "Not a Number"

    def update_wage_field(self):
         """
         Update the wage field using the provided formula, calculating luv over the Moore neighborhood.
         ωuv(t) = (Ā/2) * [1 + exp(a1 * luv(t) + a2 * (luv(t))^2)] * 0.5 * (luv(t))^(-0.5) +
                   (Ā/2) * exp(a1 * luv(t) + a2 * (luv(t))^2) * (a1 + 2 * a2 * luv(t)) * (luv(t))^0.5
         """
         min_wage = 0.1  # Example minimum wage value

         for u in range(self.grid.width):
             for v in range(self.grid.height):
                 # Get Moore neighborhood for the current cell
                 neighborhood = self.grid.get_neighborhood((u, v), moore=True, include_center=True)  # Include center!

                 total_employed_in_neighborhood = 0
                 neighborhood_size = len(neighborhood)

                 for cell in neighborhood:
                     total_employed_in_neighborhood += sum(1 for a in self.grid.get_cell_list_contents(cell) if a.is_employed)

                 # Calculate local labor density (luv) over the neighborhood
                 luv = total_employed_in_neighborhood / neighborhood_size if neighborhood_size > 0 else 0

                 # Apply the wage formula
                 if luv > 0:  # Avoid division by zero
                     term1 = (self.A / 2) * (1 + np.exp(self.a1 * luv + self.a2 * (luv**2))) * 0.5 * (luv**(-0.5))
                     term2 = (self.A / 2) * np.exp(self.a1 * luv + self.a2 * (luv**2)) * (self.a1 + 2 * self.a2 * luv) * (luv**0.5)
                     self.wage_field[u, v] = term1 + term2
                 else:
                     self.wage_field[u, v] = min_wage

import random
from agents import ACFAgent
from mesa.space import SingleGrid

class ACFMigrationModel2(mesa.Model):
    """
    Model of migration and economic agglomeration
    """
    def __init__(self, num_agents=80, width=10, height=10, hiring_rate=0.1, firing_rate=0.05, movement=True,
                 A=1.0, a1=0.1, a2=-0.01, initial_agent_locations=None, initial_wage_field=None):  # Initialize parameters
        super().__init__()
        self.num_agents = num_agents
        self.grid = SingleGrid(width, height, torus=True)
        self.hiring_rate = hiring_rate
        self.firing_rate = firing_rate
        self.running = True
        self.movement = movement

        self.A = A # This is a constant parameter related to the base productivity when cooperative effects are negligibl
        self.a1 = a1
        self.a2 = a2
        self.beta = 0.7 #Given beta value

        # Initialize agents - Half employed, half unemployed
        for i in range(self.num_agents):
            is_employed = i < num_agents / 2
            if initial_agent_locations:
                agent_location = initial_agent_locations[i]  # Use provided locations
            else:
                 agent_location = random.choice(list(self.grid.coord_iter()))
            a = ACFAgent(i, self, agent_location, random.randint(1, 5), random.uniform(15, 25),
                           random.uniform(0.1, 0.5),
                           aspiration_factor=0.2, capability_factor=0.3,
                           is_employed=is_employed)
            self.grid.place_agent(a, agent_location)


        # Initialize wage field (2D array)
        if initial_wage_field is not None:
            self.wage_field = initial_wage_field
        else:
            self.wage_field = np.zeros((width, height))# Initialize with zeros

        # For collecting data
        self.datacollector = mesa.DataCollector(
            model_reporters={"Employed": lambda m: sum(1 for a in m.agents if a.is_employed),
                             "Unemployed": lambda m: sum(1 for a in m.agents if not a.is_employed),
                             "Employed_Share": lambda m: sum(1 for a in m.agents if a.is_employed) / m.num_agents,
                             "Unemployed_Share": lambda m: sum(1 for a in m.agents if not a.is_employed) / m.num_agents
                             },
            agent_reporters={"IsEmployed": "is_employed"}  # Example agent reporter
        )

    def step(self):
        """Advance the model by one step."""
        self.agents.shuffle_do("step")

        # Update wage field based on agent locations and employment.
        self.update_wage_field()
        self.datacollector.collect(self)

    def get_employed_ratio(self):
        """
        Calculates the ratio of employed agents to the total number of agents.

        Returns:
            float: The ratio of employed agents to the total number of agents.
                   Returns 0 if there are no agents to avoid errors.
        """
        try:
            return sum(1 for a in self.agents if a.is_employed) / self.num_agents
        except ZeroDivisionError:
            return 0  # Or you might want to return float('nan') for "Not a Number"

    def update_wage_field(self):
         """
         Update the wage field using the provided formula, calculating luv over the Moore neighborhood.
         ωuv(t) = (Ā/2) * [1 + exp(a1 * luv(t) + a2 * (luv(t))^2)] * 0.5 * (luv(t))^(-0.5) +
                   (Ā/2) * exp(a1 * luv(t) + a2 * (luv(t))^2) * (a1 + 2 * a2 * luv(t)) * (luv(t))^0.5
         """
         min_wage = 0.1  # Example minimum wage value

         for u in range(self.grid.width):
             for v in range(self.grid.height):
                 # Get Moore neighborhood for the current cell
                 neighborhood = self.grid.get_neighborhood((u, v), moore=True, include_center=True)  # Include center!

                 total_employed_in_neighborhood = 0
                 neighborhood_size = len(neighborhood)

                 for cell in neighborhood:
                     total_employed_in_neighborhood += sum(1 for a in self.grid.get_cell_list_contents(cell) if a.is_employed)

                 # Calculate local labor density (luv) over the neighborhood
                 luv = total_employed_in_neighborhood / neighborhood_size if neighborhood_size > 0 else 0

                 # Apply the wage formula
                 if luv > 0:  # Avoid division by zero
                     term1 = (self.A / 2) * (1 + np.exp(self.a1 * luv + self.a2 * (luv**2))) * 0.5 * (luv**(-0.5))
                     term2 = (self.A / 2) * np.exp(self.a1 * luv + self.a2 * (luv**2)) * (self.a1 + 2 * self.a2 * luv) * (luv**0.5)
                     self.wage_field[u, v] = term1 + term2
                 else:
                     self.wage_field[u, v] = min_wage
