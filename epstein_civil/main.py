from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QSlider, QPushButton, QHBoxLayout, QLineEdit, QGraphicsView, QGraphicsScene
from PyQt5.QtCore import Qt

from mesa.examples.advanced.epstein_civil_violence.agents import Citizen, CitizenState, Cop
from mesa.examples.advanced.epstein_civil_violence.model import EpsteinCivilViolence
from mesa.visualization import make_plot_component, make_space_component

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
    "citizen_density": {
        "label": "Initial Agent Density",
        "value": 0.7,
        "min": 0.0,
        "max": 0.9,
        "step": 0.1
    },
    "cop_density": {
        "label": "Initial Cop Density",
        "value": 0.04,
        "min": 0.0,
        "max": 0.1,
        "step": 0.01
    },
    "citizen_vision": {
        "label": "Citizen Vision",
        "value": 7,
        "min": 1,
        "max": 10,
        "step": 1
    },
    "cop_vision": {
        "label": "Cop Vision",
        "value": 7,
        "min": 1,
        "max": 10,
        "step": 1
    },
    "legitimacy": {
        "label": "Government Legitimacy",
        "value": 0.82,
        "min": 0.0,
        "max": 1,
        "step": 0.01
    },
    "max_jail_term": {
        "label": "Max Jail Term",
        "value": 30,
        "min": 0,
        "max": 50,
        "step": 1
    }
}

space_component = make_space_component(
    citizen_cop_portrayal, post_process=post_process, draw_grid=False
)

chart_component = make_plot_component(
    {state.name.lower(): agent_colors[state] for state in CitizenState}
)

epstein_model = EpsteinCivilViolence()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Epstein Civil Violence")
        
        main_layout = QVBoxLayout()
        
        # Controls Section
        controls_layout = QVBoxLayout()
        controls_layout.addWidget(QLabel("Play Interval (ms)"))
        play_interval_slider = QSlider(Qt.Horizontal)
        controls_layout.addWidget(play_interval_slider)
        
        controls_layout.addWidget(QLabel("Render Interval (steps)"))
        render_interval_slider = QSlider(Qt.Horizontal)
        controls_layout.addWidget(render_interval_slider)
        
        buttons_layout = QHBoxLayout()
        reset_button = QPushButton("RESET")
        play_button = QPushButton("PLAY")
        step_button = QPushButton("STEP")
        buttons_layout.addWidget(reset_button)
        buttons_layout.addWidget(play_button)
        buttons_layout.addWidget(step_button)
        controls_layout.addLayout(buttons_layout)
        
        main_layout.addLayout(controls_layout)
        
        # Model Parameters Section
        params_layout = QVBoxLayout()
        
        # Seed input
        seed_label = QLabel(model_params["seed"]["label"])
        seed_input = QLineEdit(str(model_params["seed"]["value"]))
        params_layout.addWidget(seed_label)
        params_layout.addWidget(seed_input)
        
        # Sliders for other parameters
        self.slider_labels = {}
        for param, slider in model_params.items():
            if param not in ["seed", "height", "width"]:
                params_layout.addWidget(QLabel(slider["label"]))
                slider_widget = QSlider(Qt.Horizontal)
                slider_widget.setMinimum(int(slider["min"] * 100))
                slider_widget.setMaximum(int(slider["max"] * 100))
                slider_widget.setValue(int(slider["value"] * 100))
                slider_label = QLabel(f"{slider['value']}")
                slider_widget.valueChanged.connect(lambda value, label=slider_label: label.setText(f"{value / 100:.2f}"))
                params_layout.addWidget(slider_widget)
                params_layout.addWidget(slider_label)
                self.slider_labels[param] = slider_label
        
        main_layout.addLayout(params_layout)
        
        # Information Section
        info_layout = QVBoxLayout()
        self.step_label = QLabel("Step: 0")
        info_layout.addWidget(self.step_label)
        
        main_layout.addLayout(info_layout)
        
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)
        
        # Create grid window
        self.grid_window = QMainWindow()
        self.grid_window.setWindowTitle("Grid Visualization")
        self.grid_scene = QGraphicsScene()
        self.grid_view = QGraphicsView(self.grid_scene)
        self.grid_window.setCentralWidget(self.grid_view)
        self.grid_window.setGeometry(self.geometry())  # Set the same geometry as the main window
        self.grid_window.show()
        
        # Draw initial grid
        self.draw_grid(model_params["height"], model_params["width"])
        
        # Connect buttons to actions
        reset_button.clicked.connect(self.reset_action)
        play_button.clicked.connect(self.play_action)
        step_button.clicked.connect(self.step_action)
    
    def reset_action(self):
        print("Reset button clicked")
        # Add reset logic here
    
    def play_action(self):
        print("Play button clicked")
        epstein_model.run_model()
        self.update_step_label()
    
    def step_action(self):
        print("Step button clicked")
        epstein_model.step()
        self.update_step_label()
    
    def update_step_label(self):
        current_step = epstein_model.steps
        self.step_label.setText(f"Step: {current_step}")
    
    def draw_grid(self, height, width):
        self.grid_scene.clear()
        for i in range(height):
            for j in range(width):
                self.grid_scene.addRect(j * 10, i * 10, 10, 10)

app = QApplication([])
window = MainWindow()
window.show()
app.exec_()
