import mesa
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

# --- Default Parameters ---
dflt_G = nx.DiGraph()
dflt_G.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3)])
dflt_link_parameters = [(0.1, 0.0), (0.0, 15.0), (0.0, 15.0), (0.1, 0.0)]
dflt_T = 50.0

def link_path_adjacency(link, path):
    """Check if a given link (u, v) is on a given path."""
    for i in range(len(path) - 1):
        if (link[0] == path[i] and link[1] == path[i+1]):
            return 1
    return 0

def create_properties(G, link_parameters, T):
    """Create a dictionary of model properties."""
    # Define demand-delay functions for each link
    demand_delay_functions = {i: (lambda x, i=i: link_parameters[i][0] * x + link_parameters[i][1]) for i in range(len(link_parameters))}

    # Get all paths from node 0 to node 3
    paths = list(nx.all_simple_paths(G, source=0, target=3))

    # Calculate link-path adjacency matrix
    edges = sorted(list(G.edges()))
    A = np.zeros((len(edges), len(paths)), dtype=int)
    for i, edge in enumerate(edges):
        for j, path in enumerate(paths):
            A[i, j] = link_path_adjacency(edge, path)

    # Initially route times are set to zero
    route_times = np.zeros(len(paths))

    return {
        'T': T,
        'link_path_matrix': A,
        'demand_delay': demand_delay_functions,
        'route_times': route_times,
        'routes': paths
    }

class DriverAgent(mesa.Agent):
    """An agent representing a driver."""
    def __init__(self, model):
        super().__init__(model)
        self.route_choice = len(self.model.routes)-1 # Initially, all agents choose the last route of nx.all_simple_paths
        self.travel_time = 0.0

    def step(self):
        """Agent's step action."""
        T = self.model.T
        previous_route_times = self.model.route_times
        nroutes = len(previous_route_times)
        
        # Current route's travel time from the previous day
        t = previous_route_times[self.route_choice]
        
        # Choose a random alternative route
        alternative_routes = list(range(nroutes))
        alternative_routes.remove(self.route_choice)
        if not alternative_routes:
            return # No alternative routes to choose from
            
        alternative_route = random.choice(alternative_routes)
        t_alternative = previous_route_times[alternative_route]
        
        # Decide whether to switch routes
        delta_t = t_alternative - t
        if delta_t <= 0:
            prob_switch = 1.0 - np.exp(delta_t / T)
            if self.random.random() < prob_switch:
                self.route_choice = alternative_route

class CGModel(mesa.Model):
    """The Congestion Game model."""
    def __init__(self, num_agents=100, G=dflt_G, link_parameters=dflt_link_parameters, T=dflt_T, seed=None):
        super().__init__(seed=seed)
        properties = create_properties(G, link_parameters, T)
        self.T = properties['T']
        self.link_path_matrix = properties['link_path_matrix']
        self.demand_delay = properties['demand_delay']
        self.route_times = properties['route_times']
        self.routes = properties['routes']
        
        self.num_agents = num_agents
        
        # Create agents
        DriverAgent.create_agents(self, num_agents)

        self.datacollector = mesa.DataCollector(
            agent_reporters={"route_choice": "route_choice", "travel_time": "travel_time"}
        )

    def step(self):
        """Model's step action."""
        self.agents.do("step")
        
        A = self.link_path_matrix
        nlinks, npaths = A.shape
        
        # Count agents on each route
        path_counts = np.zeros(npaths, dtype=int)
        for agent in self.agents:
            path_counts[agent.route_choice] += 1
            
        # Calculate link counts
        link_counts = A @ path_counts
        
        # Calculate link travel times
        link_travel_times = np.array([self.demand_delay[i](link_counts[i]) for i in range(nlinks)])
        
        # Calculate updated path travel times
        self.route_times = link_travel_times @ A
        
        # Update each agent's travel time for their chosen route
        for agent in self.agents:
            agent.travel_time = self.route_times[agent.route_choice]
            
        self.datacollector.collect(self)

def run_model(num_agents=100, num_steps=100, G=dflt_G, link_parameters=dflt_link_parameters, T=dflt_T):
    """Run the model and return the collected data and plots."""
    model = CGModel(num_agents=num_agents, G=G, link_parameters=link_parameters, T=T)
    for _ in range(num_steps):
        model.step()
        
    data = model.datacollector.get_agent_vars_dataframe()
    
    # Plot average travel time
    avg_travel_time = data.groupby('Step')['travel_time'].mean()
    fig1, ax1 = plt.subplots()
    avg_travel_time.plot(ax=ax1, ylim=(0, 50))
    ax1.set_title("Average Travel Time")
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Time")
    ax1.grid(axis='both', color='0.5', linestyle='--', linewidth=1)
    
    # Plot agent counts per route
    route_counts = data.groupby(['Step', 'route_choice']).size().unstack(fill_value=0)

    fig2, ax2 = plt.subplots()
    for i, path in enumerate(model.routes):
        route_counts[i].plot(ax=ax2, label=f"Route {i+1}: {path}")
    ax2.set_title("Route Choices Over Time")
    ax2.set_xlabel("Steps")
    ax2.set_ylabel("Users")
    ax2.grid(axis='both', color='0.5', linestyle='--', linewidth=1)
    ax2.legend()
    
    plt.tight_layout()
    return data, fig1, fig2

if __name__ == '__main__':
    # G2 = nx.DiGraph()
    # G2.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3), (1, 2)])
    # linkparams2 = [(0.1, 0.0),
    #                (0.0, 15.0),
    #                (0.0, 5.0),  # (1->2)
    #                (0.0, 15.0),
    #                (0.1, 0.0)]
    data, fig1, fig2 = run_model()
    plt.show()




