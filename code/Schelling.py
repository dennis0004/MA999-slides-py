import mesa
import numpy as np
import matplotlib.pyplot as plt

class SchellingAgent(mesa.Agent):
    """
    Schelling segregation agent.
    """
    def __init__(self, model, agent_type):
        super().__init__(model)
        self.type = agent_type
        self.mood = False
        self.neighbours = 0
        self.interfaces = 0.0

    def step1(self):
        """
        Decide to move or not.
        """
        similar = 0
        neighbors = self.model.grid.get_neighbors(self.pos, moore=True, include_center=False)
        
        if not neighbors:
            self.mood = False
            if self.model.grid.exists_empty_cells():
                self.model.grid.move_to_empty(self)
            return

        for neighbor in neighbors:
            if neighbor.type == self.type:
                similar += 1

        if similar < self.model.min_to_be_happy:
            self.mood = False
            if self.model.grid.exists_empty_cells():
                self.model.grid.move_to_empty(self)
        else:
            self.mood = True

    def step2(self):
        """
        Calculate interface density.
        """
        neighbors = self.model.grid.get_neighbors(self.pos, moore=True, include_center=False)
        count_total_neighbours = len(neighbors)
        
        if count_total_neighbours == 0:
            self.neighbours = 0
            self.interfaces = 0.0
            return

        count_neighbors_other_group = 0
        for neighbor in neighbors:
            if self.type != neighbor.type:
                count_neighbors_other_group += 1

        self.neighbours = count_total_neighbours
        self.interfaces = count_neighbors_other_group / count_total_neighbours


class Schelling(mesa.Model):
    """
    Model class for the Schelling segregation model.
    """
    def __init__(self, width=20, height=20, num_agents=320, minority_pc=0.5, min_to_be_happy=3, seed=125):
        super().__init__(seed=seed)
        self.width = width
        self.height = height
        self.num_agents = num_agents
        self.minority_pc = minority_pc
        self.min_to_be_happy = min_to_be_happy
        self.grid = mesa.space.SingleGrid(width, height, torus=False)

        # Create agents
        SchellingAgent.create_agents(self, num_agents, 0)
        num_group1 = int(self.num_agents * self.minority_pc)
        for i in range(self.num_agents):
            agent_type = 1 if i < num_group1 else 2
            self.agents[i].type = agent_type
            self.grid.move_to_empty(self.agents[i])

        # Initial calculation of neighbours and interfaces for all agents
        self.agents.do("step2")

        model_reporters = {
            "mean_interfaces": lambda m: np.mean([a.interfaces for a in m.agents if a.neighbours > 0] or [0]),
            "sum_mood": lambda m: np.sum([a.mood for a in m.agents]),
            "mean_neighbours": lambda m: np.mean([a.neighbours for a in m.agents] or [0]),
        }
        self.datacollector = mesa.datacollection.DataCollector(model_reporters=model_reporters)
        self.datacollector.collect(self)

    def step(self):
        """
        Run one step of the model.
        """
        self.agents.do("step1")
        self.agents.do("step2")
        self.datacollector.collect(self)

if __name__ == '__main__':
    model = Schelling(width=20, height=20, num_agents=320, minority_pc=0.5, min_to_be_happy=3, seed=125)
    for i in range(50):
        model.step()

    data = model.datacollector.get_model_vars_dataframe()
    plt.plot(data['mean_interfaces'][1:], label=f'min_to_be_happy = 3')
    plt.ylim(0, 1)
    plt.xlim(0, 50)
    plt.title('Interface density')
    plt.xlabel('Steps')
    plt.ylabel('Fraction of out-group neighbours')
    plt.grid(axis='both', color='0.9', linestyle='--', linewidth=1)
    plt.legend()
    plt.show()




