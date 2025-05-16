import mesa
import numpy as np

class ABP(mesa.Agent):
    """
    An agent with an internal state (employed or unemployed) that moves
    according to the active Brownian particle model.
    """

    def __init__(self, unique_id, model, is_employed=False):
        super().__init__(model)
        self.is_employed = is_employed
        self.speed = 1  # Base speed
        self.wage_sensitivity = 0.1

    def move(self):
        if not self.is_employed:  # Only unemployed agents move
            # Get neighborhood
            neighbors = self.model.grid.get_neighborhood(
                self.pos,
                moore=True,  # Use Moore neighborhood (8 cells)
                include_center=False)

            # Calculate wage gradient using the updated wage field
            wage_gradient_x = 0
            wage_gradient_y = 0
            for neighbor_pos in neighbors:
                wage_gradient_x += self.model.wage_field[neighbor_pos[0], neighbor_pos[1]] - self.model.wage_field[self.pos[0], self.pos[1]]
                wage_gradient_y += self.model.wage_field[neighbor_pos[0], neighbor_pos[1]] - self.model.wage_field[self.pos[0], self.pos[1]]

            # Movement influenced by wage gradient and randomness
            dx = wage_gradient_x * self.wage_sensitivity + np.random.uniform(-1, 1)
            dy = wage_gradient_y * self.wage_sensitivity + np.random.uniform(-1, 1)

            new_x = self.pos[0] + int(dx * self.speed)
            new_y = self.pos[1] + int(dy * self.speed)

            # Wrap around edges
            new_x = new_x % self.model.grid.width
            new_y = new_y % self.model.grid.height
            
            original_pos = self.pos  # Store the original position
            new_pos = (new_x, new_y)
            
            attempted_positions = [new_pos] # List of positions already tried
            
            while not self.model.grid.is_cell_empty(new_pos):
                # If the new position is not empty, try a random neighbor
                
                valid_neighbors = [
                    (x, y) for x, y in neighbors
                    if (x % self.model.grid.width, y % self.model.grid.height) not in attempted_positions
                ]
                
                if not valid_neighbors:
                    new_pos = original_pos #Stay in the same position
                    break # All neighbors are occupied
                
                new_pos = self.random.choice(valid_neighbors)
                attempted_positions.append(new_pos)
                
            if new_pos != original_pos:
              self.model.grid.move_agent(self, new_pos)

    def update_employment_status(self):
        # Simplified hiring/firing.  These rates could depend on local conditions.
        if self.is_employed:
            if np.random.random() < self.model.firing_rate:
                self.is_employed = False
        else:
            if np.random.random() < self.model.hiring_rate:
                self.is_employed = True

    def step(self):
        self.move()
        self.update_employment_status()

class ACFAgent(ABP):  # ACFAgent now inherits from ABP
    """
    An agent with attributes and decision-making based on de Haas' aspirations and capabilities framework,
    inheriting movement from ABP.
    """
    def __init__(self, unique_id, model, location, skill, wage_expect, decision_threshold,
                 aspiration_factor, capability_factor, is_employed=False):
        super().__init__(unique_id, model, is_employed)  # Call ABP's constructor
        self.location = location
        self.skill = skill
        self.wage_expect = wage_expect
        self.decision_threshold = decision_threshold
        self.aspiration_factor = aspiration_factor
        self.capability_factor = capability_factor
        self.speed = 1
        self.wage_sensitivity = 0.1

    def update_wage_expect(self):
        """
        Aspirations are influenced by the average wages in the agent's current location.
        """
        self.wage_expect = self.wage_expect + self.aspiration_factor * (self.model.wage_field[self.pos[0], self.pos[1]] - self.wage_expect)

    def make_decision(self):
        """
        Agent decision-making based on aspirations and capabilities.
        """
        # Get the wages in the agent's current location
        potential_wages = self.model.wage_field[self.pos[0], self.pos[1]]

        # Get neighbor wages
        neighbors = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
        for neighbor_pos in neighbors:
            potential_wages = max(potential_wages, self.model.wage_field[neighbor_pos[0], neighbor_pos[1]])

        wage_differential = potential_wages - self.wage_expect
        decision_value = (self.aspiration_factor * wage_differential) + (self.capability_factor * self.skill)

        if decision_value > self.decision_threshold:
            return True
        else:
            return False

    def step(self):
        self.move()  # Inherit movement from ABP
        self.update_employment_status()
        self.update_wage_expect()