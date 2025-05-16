import solara
from mesa.examples.advanced.migration.agents import (
    ABP
)
import matplotlib.pyplot as plt
import numpy as np
from mesa.examples.advanced.migration.model import MigrationModel
from mesa.visualization import (
    Slider,
    SolaraViz,
    make_space_component,
    make_plot_component
)


def agent_portrayal(agent):
    return {
        "color": "green" if agent.is_employed == True else "red",
        "marker": "s",
        "size": 50,
    }
def post_process_lineplot(ax):
    ax.set_ylim(ymin=0)
    ax.set_ylabel("Ratio")
    ax.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")

def post_process(ax):
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

def post_process_heatmap(ax):
    ax.set_aspect("equal")
    ax.set_title("Employed Agent Density")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")


def get_employed_ratio_text(model):
    """
    Calculates and formats the employed ratio for display.

    Args:
        model (MigrationModel): The current Mesa model instance.

    Returns:
        solara.Markdown: A Solara Markdown element containing the formatted ratio.
    """
    employed_count = model.get_employed_ratio()
    total_agents = model.num_agents  # Assuming model has num_agents
    try:
        ratio = employed_count / total_agents
    except ZeroDivisionError:
        ratio = float('inf')  # Or you might want to use NaN: float('nan')

    ratio_text = r"$\infty$" if ratio == float('inf') else f"{ratio:.2f}"
    total_text = str(total_agents)

    return solara.Markdown(
        f"Employed Ratio (Employed/Total): {ratio_text}<br>Total Agents: {total_text}"
    )

def employed_density_plot(model):
    """
    Creates a density plot of employed agent locations.

    Args:
        model (MigrationModel): The current Mesa model instance.

    Returns:
        Any: A Solara component containing the density plot.
    """

    employed_positions = [
        (a.pos[0], a.pos[1]) for a in model.agents if a.is_employed
    ]
    if not employed_positions:
        return solara.Markdown("No employed agents to display.")  # Handle empty case

    x, y = zip(*employed_positions)

    fig, ax = plt.subplots(figsize=(6, 6))  # Adjust figsize as needed

    # Create a 2D histogram (density approximation)
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=10, density=True)  # Adjust bins as needed
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    # Display the heatmap
    ax.imshow(heatmap.T, extent=extent, origin='lower', cmap='viridis', aspect='auto')
    fig.colorbar(img, ax=ax, label='Density')
    plt.show()

model_params = {
    "seed": {
        "type": "InputText",
        "value": 777,
        "label": "Random Seed",
    },
    "height": 10,
    "width": 10,
    "hiring_rate": Slider("Initial Hiring Rate", 0.05, 0.1, 0.15, 0.2),
    "firing_rate": Slider("Initial Firing Rate", 0.025, 0.05, 0.075),
}

space_component = make_space_component(
    agent_portrayal, post_process=post_process, draw_grid=False
)

line_plot = make_plot_component(
    {"Employed": "tab:green", "Unemployed": "tab:red"},
    post_process=post_process_lineplot,
)

heatmap_plot = make_plot_component(
    {"Employed": "tab:green"},
    post_process=post_process,
)

migration_model = MigrationModel()
page = SolaraViz(
    migration_model,
    components=[space_component, line_plot, heatmap_plot],
    model_params=model_params,
    name="APB Migration",
)
page  # noqa