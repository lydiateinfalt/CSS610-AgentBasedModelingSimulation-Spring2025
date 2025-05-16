# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 20:53:22 2025

@author: petit
"""

from mesa.examples.advanced.epstein_civil_violence.agents import (
    Citizen,
    CitizenState,
    Cop,
)
from mesa.examples.advanced.epstein_civil_violence.model import EpsteinCivilViolence
from mesa.visualization import (
    Slider,
    SolaraViz,
    make_plot_component,
    make_space_component,
)

COP_COLOR = "#000000"

agent_colors = {
    CitizenState.ACTIVE: "#FE6100",
    CitizenState.QUIET: "#648FFF",
    CitizenState.ARRESTED: "#808080",
}


def citizen_cop_portrayal(agent):
    if agent is None:
        return

    portrayal = {
        "size": 50,
    }

    if isinstance(agent, Citizen):
        portrayal["color"] = agent_colors[agent.state]
    elif isinstance(agent, Cop):
        portrayal["color"] = COP_COLOR

    return portrayal


def post_process(ax):
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.get_figure().set_size_inches(10, 10)


model_params = {
    "seed": {
        "type": "InputText",
        "value": 42,
        "label": "Random Seed",
    },
    "height": 40,
    "width": 40,
    "citizen_density": Slider("Initial Agent Density", 0.7, 0.0, 0.9, 0.1),
    "cop_density": Slider("Initial Cop Density", 0.04, 0.0, 0.1, 0.01),
    "citizen_vision": Slider("Citizen Vision", 7, 1, 10, 1),
    "cop_vision": Slider("Cop Vision", 7, 1, 10, 1),
    "legitimacy": Slider("Government Legitimacy", 0.82, 0.0, 1, 0.01),
    "max_jail_term": Slider("Max Jail Term", 30, 0, 50, 1),
}

space_component = make_space_component(
    citizen_cop_portrayal, post_process=post_process, draw_grid=False
)

chart_component = make_plot_component(
    {state.name.lower(): agent_colors[state] for state in CitizenState}
)

#epstein_model = EpsteinCivilViolence()

epstein_model = EpsteinCivilViolence(height=40, 
                           width=40, 
                           citizen_density=.7, 
                           cop_density=.074, 
                           citizen_vision=7, 
                           cop_vision=7, 
                           legitimacy=.8, 
                           max_jail_term=1000, 
                           initial_unemployment_rate = 0.2,
                           corruption_level = 0.1,
                           susceptible_level = 0.6,
                           max_iters=200) # cap the number of steps the model takes
epstein_model.run_model()

page = SolaraViz(
    epstein_model,
    components=[space_component, chart_component],
    model_params=model_params,
    name="Epstein Civil Violence",
)
page  # noqa
