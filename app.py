from flask_basicauth import BasicAuth
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.rnsga2 import RNSGA2
from pymoo.algorithms.moo.spea2 import SPEA2
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.problems import get_problem
from pymoo.optimize import minimize
import plotly.graph_objects as go

# Initialize the Flask app
from flask import Flask
server = Flask(__name__)

# Configure Basic Auth
server.config["BASIC_AUTH_USERNAME"] = "admin"  # Set your username
server.config["BASIC_AUTH_PASSWORD"] = "MOEAlgosTests@py"  # Set your password
server.config["BASIC_AUTH_FORCE"] = True  # Force authentication on all routes

basic_auth = BasicAuth(server)

# Initialize the Dash app
app = dash.Dash(
    __name__,
    server=server,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    serve_locally=True,
)
app.css.append_css({"external_url": "/static/css/style.css"})

# List of problems and algorithms
PROBLEMS = ["zdt1", "zdt2", "zdt3", "dtlz1", "dtlz2"]
ALGORITHMS = {
    "NSGA2": NSGA2,
    "MOEAD": MOEAD,
    "NSGA3": NSGA3,
    "R-NSGA-II": RNSGA2,
    "SPEA2": SPEA2,
}

# Layout
app.layout = dbc.Container(
    [
        html.H1("Multi-Objective Optimization Dashboard", className="text-center my-4"),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Label("Select Problem"),
                        dcc.Dropdown(
                            id="problem-dropdown",
                            options=[{"label": prob, "value": prob} for prob in PROBLEMS],
                            value="zdt1",
                        ),
                    ],
                    width=4,
                ),
                dbc.Col(
                    [
                        html.Label("Select Algorithm"),
                        dcc.Dropdown(
                            id="algorithm-dropdown",
                            options=[{"label": algo, "value": algo} for algo in ALGORITHMS.keys()],
                            value="NSGA2",
                        ),
                    ],
                    width=4,
                ),
                dbc.Col(
                    [
                        html.Label("Number of Generations"),
                        dcc.Input(id="n-gen", type="number", value=100, min=1),
                    ],
                    width=4,
                ),
            ],
            className="mb-4",
        ),
        dcc.Graph(id="optimization-plot", style={"height": "600px"}),
        dbc.Button("Run Optimization", id="run-button", color="primary", className="mt-3"),
    ],
    fluid=True,
)

# Callback for running the optimization and updating the plot
@app.callback(
    Output("optimization-plot", "figure"),
    Input("run-button", "n_clicks"),
    [Input("problem-dropdown", "value"),
     Input("algorithm-dropdown", "value"),
     Input("n-gen", "value")],
)
def update_plot(n_clicks, problem_name, algorithm_name, n_gen):
    if not n_clicks:
        # Return an empty figure before the first run
        return go.Figure()

    # Initialize the problem and algorithm
    problem = get_problem(problem_name)

    if algorithm_name == "NSGA3":
        ref_dirs = get_reference_directions("das-dennis", problem.n_obj, n_partitions=12)
        algorithm = ALGORITHMS[algorithm_name](pop_size=92, ref_dirs=ref_dirs)
    elif algorithm_name == "MOEAD":
        ref_dirs = get_reference_directions("uniform", problem.n_obj, n_partitions=12)
        algorithm = ALGORITHMS[algorithm_name](ref_dirs=ref_dirs)
    elif algorithm_name == "R-NSGA-II":
        ref_points = get_reference_directions("uniform", problem.n_obj, n_partitions=4)
        algorithm = ALGORITHMS[algorithm_name](ref_points=ref_points)
    else:
        algorithm = ALGORITHMS[algorithm_name](pop_size=50)

    # Run the optimization
    res = minimize(problem, algorithm, ("n_gen", n_gen), seed=1, verbose=False)

    # Retrieve the optimal Pareto front (if available)
    optimal_pareto_front = problem.pareto_front() if hasattr(problem, "pareto_front") else None

    # Prepare data for the plot
    pareto_front = res.F
    all_individuals = [ind.F for ind in res.pop]

    # Create the figure
    fig = go.Figure()

    if problem.n_obj == 2:
        fig.add_trace(
            go.Scatter(
                x=[ind[0] for ind in all_individuals],
                y=[ind[1] for ind in all_individuals],
                mode="markers",
                marker=dict(color="blue", size=5),
                name="All Individuals",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[ind[0] for ind in pareto_front],
                y=[ind[1] for ind in pareto_front],
                mode="markers",
                marker=dict(color="red", size=8),
                name="Pareto Front",
            )
        )
        if optimal_pareto_front is not None:
            fig.add_trace(
                go.Scatter(
                    x=optimal_pareto_front[:, 0],
                    y=optimal_pareto_front[:, 1],
                    mode="lines",
                    line=dict(color="green", width=2),
                    name="Optimal Pareto Front",
                )
            )
        fig.update_layout(
            title=f"Optimization Results: {problem_name} with {algorithm_name}",
            xaxis_title="Objective 1",
            yaxis_title="Objective 2",
        )
    elif problem.n_obj == 3:
        fig.add_trace(
            go.Scatter3d(
                x=[ind[0] for ind in all_individuals],
                y=[ind[1] for ind in all_individuals],
                z=[ind[2] for ind in all_individuals],
                mode="markers",
                marker=dict(color="blue", size=3),
                name="All Individuals",
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=[ind[0] for ind in pareto_front],
                y=[ind[1] for ind in pareto_front],
                z=[ind[2] for ind in pareto_front],
                mode="markers",
                marker=dict(color="red", size=5),
                name="Pareto Front",
            )
        )
        if optimal_pareto_front is not None:
            fig.add_trace(
                go.Scatter3d(
                    x=optimal_pareto_front[:, 0],
                    y=optimal_pareto_front[:, 1],
                    z=optimal_pareto_front[:, 2],
                    mode="lines",
                    line=dict(color="green", width=2),
                    name="Optimal Pareto Front",
                )
            )
        fig.update_layout(
            title=f"Optimization Results: {problem_name} with {algorithm_name}",
            scene=dict(
                xaxis_title="Objective 1",
                yaxis_title="Objective 2",
                zaxis_title="Objective 3",
            ),
        )
    else:
        fig.update_layout(
            title="Visualization not supported for more than 3 objectives.",
        )

    return fig


if __name__ == "__main__":
    app.run_server(debug=True)
