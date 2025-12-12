import mesa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Default Parameters ---
SIZE_DFLT = 25
T_OPT_DFLT = 22.5
T_RANGE_DFLT = 10.0
P_DEATH_DFLT = 0.1
P_GROWTH_DFLT = 0.05
DELTA_DFLT = 0.25
ALBEDO_DFLT = [0.75, 0.25, 0.5]  # white, black, empty
F_DFLT = lambda step: (T_OPT_DFLT + T_RANGE_DFLT) * 2.0

class DaisyPatch(mesa.Agent):
    """An agent representing a patch of ground in the Daisyworld model."""
    def __init__(self, model, colour, temperature):
        super().__init__(model)
        self.colour = colour  # 1: white, 2: black, 3: empty
        self.temperature = temperature

    def step(self):
        """Agent's step action."""
        if self.colour == 3:  # If empty
            # With low probability, an empty patch can grow daisies
            if self.random.random() <= self.model.p_growth:
                self.colour = self.random.choice([1, 2])
        else:  # If occupied by a daisy
            # An occupied patch can attempt to reproduce
            neighbors = self.model.grid.get_neighbors(self.pos, moore=True)
            if neighbors:
                neighbor = self.random.choice(neighbors)
                if neighbor.colour == 3:  # Reproduce to an empty patch
                    growth_prob = self.model.growth_rate(self.temperature)
                    if self.random.random() <= growth_prob:
                        neighbor.colour = self.colour
            
            # An occupied patch can die
            if self.random.random() <= self.model.p_death:
                self.colour = 3

class Daisyworld(mesa.Model):
    """The Daisyworld model."""
    def __init__(self, width=SIZE_DFLT, height=SIZE_DFLT, 
                 t_opt=T_OPT_DFLT, t_range=T_RANGE_DFLT, 
                 p_death=P_DEATH_DFLT, p_growth=P_GROWTH_DFLT, 
                 delta=DELTA_DFLT, albedo=ALBEDO_DFLT, flux_func=F_DFLT, seed=None):
        
        super().__init__(seed=seed)
        self.grid = mesa.space.SingleGrid(width, height, torus=True)
        
        self.t_opt = t_opt
        self.t_range = t_range
        self.p_death = p_death
        self.p_growth = p_growth
        self.delta = delta
        self.albedo = albedo
        self.flux = flux_func
        self.step_num = 0

        # Growth rate function
        self.growth_rate = lambda T: max(-(T - self.t_opt + self.t_range) * (T - self.t_opt - self.t_range) / (self.t_range**2), 0.0)

        # Initial temperature of a "bare" planet
        t_init = self.flux(1) * (1 - self.albedo[2]) # Absorbed energy from empty patch albedo

        # Create agents (patches)
        DaisyPatch.create_agents(self, width * height, colour=3, temperature=t_init)
        for i in range(width * height):
            x = i % width
            y = i // width
            # All patches are initially empty
            self.grid.move_agent(self.agents[i], (x, y))

        self.datacollector = mesa.DataCollector(
            agent_reporters={"colour": "colour", "temperature": "temperature"}
        )

    def step(self):
        """Model's step action."""
        self.agents.shuffle_do("step")

        # Calculate absorbed energy proportions
        absorbed_energy = [1.0 - x for x in self.albedo]
        
        # Calculate local temperatures
        current_flux = self.flux(self.step_num)
        
        # Store new temperatures to apply them all at once
        new_temperatures = {}

        for agent in self.agents:
            t1 = absorbed_energy[agent.colour - 1] * current_flux
            
            neighbors = self.grid.get_neighbors(agent.pos, moore=True)
            if neighbors:
                t2 = sum(absorbed_energy[n.colour - 1] * current_flux for n in neighbors) / len(neighbors)
            else:
                t2 = 0.0
            
            # Weighted sum for new local temperature
            new_temp = (1.0 - self.delta) * t1 + self.delta * t2
            new_temperatures[agent.unique_id] = new_temp

        # Update all agent temperatures
        for agent in self.agents:
            agent.temperature = new_temperatures[agent.unique_id]

        self.step_num += 1
        self.datacollector.collect(self)

def run_daisyworld(n_steps, **kwargs):
    """Run the Daisyworld model and return data and a plot."""
    model = Daisyworld(**kwargs)
    for _ in range(n_steps):
        model.step()
        
    data = model.datacollector.get_agent_vars_dataframe()
    
    # --- Analysis and Plotting ---
    gdf = data.groupby('Step')
    df_t = gdf['temperature'].mean().reset_index().rename(columns={'temperature': 'T_average'})
    
    colour_counts = data.groupby(['Step', 'colour']).size().unstack(fill_value=0)
    total_patches = model.grid.width * model.grid.height
    
    fig, ax = plt.subplots()
    
    # Plot average temperature
    ax.plot(df_t['Step'], df_t['T_average'] / model.t_opt, label="T/T_opt", color="red")
    
    # Plot daisy proportions
    if 1 in colour_counts.columns:
        ax.plot(colour_counts.index, colour_counts[1] / total_patches, label="Proportion of white daisies", color="lightgrey")
    if 2 in colour_counts.columns:
        ax.plot(colour_counts.index, colour_counts[2] / total_patches, label="Proportion of black daisies", color="black")
    if 3 in colour_counts.columns:
        ax.plot(colour_counts.index, colour_counts[3] / total_patches, label="Proportion of empty patches", color="brown")
        
    ax.set_xlabel("Steps")
    ax.set_ylabel("Proportion / Normalized Temperature")
    ax.legend()
    plt.tight_layout()
    
    return data, fig

if __name__ == '__main__':
    data, fig = run_daisyworld(50)
    plt.show()
# %%



