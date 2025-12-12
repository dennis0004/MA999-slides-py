import mesa
import matplotlib.pyplot as plt
import numpy as np

class CA1DAgent(mesa.Agent):
    """An agent in a 1D cellular automaton."""

    def __init__(self, model, state=0):
        super().__init__(model)
        self.state = state
        self.next_state = None

    def step(self):
        """Calculates the next state based on the Wolfram rule."""
        x, y = self.pos

        left_pos = self.model.grid.torus_adj((x - 1, y))
        right_pos = self.model.grid.torus_adj((x + 1, y))

        left_agent = self.model.grid.get_cell_list_contents([left_pos])[0]
        right_agent = self.model.grid.get_cell_list_contents([right_pos])[0]

        rule_index_str = str(left_agent.state) + str(self.state) + str(right_agent.state)

        self.next_state = self.model.rules[rule_index_str]

    def advance(self):
        """Sets the new state."""
        self.state = self.next_state


class CA1DModel(mesa.Model):
    """A 1D cellular automaton model."""

    def __init__(self, gridsize, rules, seed=None, initial_condition="singleton"):
        super().__init__(seed=seed)
        self.num_agents = gridsize
        self.grid = mesa.space.SingleGrid(gridsize, 1, torus=True)
        self.rules = rules

        # Create agents
        CA1DAgent.create_agents(self, self.num_agents, state=0)
        for i in range(self.num_agents):
            self.grid.place_agent(self.agents[i], (i, 0))

        if initial_condition == "singleton":
            self.agents[self.num_agents - 1].state = 1
        elif initial_condition == "random":
            for agent in self.agents:
                agent.state = self.random.choice([0, 1])

    def step(self):
        """Executes one model step."""
        self.agents.do("step")
        self.agents.do("advance")

def run(rules, gridsize, nsteps, initial_condition, update="synchronous"):
    model = CA1DModel(gridsize=gridsize, rules=rules, initial_condition=initial_condition)

    data = np.zeros((nsteps + 1, gridsize), dtype=int)

    sorted_agents = sorted(model.agents, key=lambda x: x.unique_id)
    data[0, :] = [agent.state for agent in sorted_agents]

    for i in range(nsteps):
        if update == "synchronous":
            model.step()
        else: # sequential
            for agent in model.agents:
                agent.step()
                agent.advance()

        sorted_agents = sorted(model.agents, key=lambda x: x.unique_id)
        data[i + 1, :] = [agent.state for agent in sorted_agents]

    fig, ax = plt.subplots()
    ax.imshow(data, cmap='gray_r', interpolation='nearest')
    ax.set_aspect('equal')
    ax.axis('off')

    return fig, data

if __name__ == '__main__':
    rule110 = {"111": 0, "110": 1, "101": 1, "100": 0, "011": 1,
               "010": 1, "001": 1, "000": 0}
    rule22 = {"111": 0, "110": 0, "101": 0, "100": 1, "011": 0,
              "010": 1, "001": 1, "000": 0}

    n = 11
    model = CA1DModel(gridsize=n, rules=rule110, initial_condition="singleton")
    print(f"Model initialized: {model}")

    steps = round(n / 2)
    p, data = run(rules=rule22, gridsize=n, initial_condition="singleton", nsteps=steps, update="synchronous")
    print(data)

    p, data = run(rules=rule22, gridsize=n, initial_condition="singleton", nsteps=steps, update="sequential")
    print(data)




