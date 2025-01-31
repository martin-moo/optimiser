# app.py
from flask import Flask, render_template, request, jsonify, send_file
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.rnsga2 import RNSGA2
from pymoo.algorithms.moo.spea2 import SPEA2
from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
import numpy as np
import pandas as pd
import time
import json
import io
import csv
from pymoo.decomposition.tchebicheff import Tchebicheff
from pymoo.decomposition.weighted_sum import WeightedSum
from pymoo.decomposition.pbi import PBI
from pymoo.indicators.gd import GD
from pymoo.indicators.hv import HV
from pymoo.core.problem import ElementwiseProblem
import ast
import sympy
from sympy.parsing.sympy_parser import parse_expr
from sympy.utilities.lambdify import lambdify
import traceback
from sympy import Symbol, sympify, lambdify
from pymoo.core.problem import ElementwiseProblem
import re


app = Flask(__name__)

# Dictionary to store benchmark problems
BENCHMARK_PROBLEMS = {
    'zdt1': 'ZDT1',
    'zdt2': 'ZDT2',
    'zdt3': 'ZDT3',
    'dtlz1': 'DTLZ1',
    'dtlz2': 'DTLZ2'
}

@app.route('/')
def index():
    algorithms = ['NSGA2', 'NSGA3', 'MOEAD', 'RNSGA2', 'SPEA2']
    return render_template('index.html', 
                         algorithms=algorithms,
                         problems=list(BENCHMARK_PROBLEMS.keys()))

@app.route('/optimize/benchmark', methods=['POST'])
def optimize_benchmark():
    try:
        start_time = time.time()
        data = request.json
        algorithm_name = data['algorithm']
        problem_name = data['problem']
        
        # Adjust population size and generations for MOEAD
        if algorithm_name == 'MOEAD':
            pop_size = data.get('population_size', 50)  # Reduced from 100
            n_gen = data.get('n_gen', 50)  # Reduced from 100
        else:
            pop_size = data.get('population_size', 100)
            n_gen = data.get('n_gen', 100)
        
        # Get problem instance
        problem = get_problem(problem_name)
        optimal_pareto_front = problem.pareto_front()
        
        # Initialize algorithm based on type
        if algorithm_name == 'NSGA2':
            algorithm = NSGA2(pop_size=pop_size, n_offsprings=pop_size)
        elif algorithm_name == 'SPEA2':
            algorithm = SPEA2(pop_size=pop_size, n_offsprings=pop_size)
        elif algorithm_name == 'NSGA3':
            n_partitions = data.get('n_partitions', 70)
            ref_dirs = get_reference_directions("das-dennis", problem.n_obj, n_partitions=n_partitions)
            algorithm = NSGA3(pop_size=pop_size, ref_dirs=ref_dirs)
        elif algorithm_name == 'MOEAD':
            n_partitions = 12  # Reduced number of partitions for faster execution
            ref_dirs = get_reference_directions("das-dennis", problem.n_obj, n_partitions=n_partitions)
            n_neighbors = min(data.get('n_neighbors', 15), len(ref_dirs) - 1)  # Reduced number of neighbors
            
            decomposition_name = data.get('decomposition', 'tchebicheff')
            if decomposition_name == 'tchebicheff':
                decomposition = Tchebicheff()
            elif decomposition_name == 'weighted_sum':
                decomposition = WeightedSum()
            elif decomposition_name == 'pbi':
                decomposition = PBI()
            else:
                return jsonify({'success': False, 'error': f'Unknown decomposition method: {decomposition_name}'}), 400
            
            algorithm = MOEAD(
                ref_dirs=ref_dirs,
                n_neighbors=n_neighbors,
                decomposition=decomposition,
                prob_neighbor_mating=data.get('prob_neighbor_mating', 0.7)
            )
        elif algorithm_name == 'RNSGA2':
            # Adjust reference points based on the problem
            if problem_name in ['dtlz1', 'dtlz2']:
                ref_points = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]])
            else:
                ref_points = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
            
            algorithm = RNSGA2(
                ref_points=ref_points,
                pop_size=pop_size,
                epsilon=data.get('epsilon', 0.01),
                normalization='front'
            )
        else:
            return jsonify({'success': False, 'error': f'Unknown algorithm: {algorithm_name}'}), 400

        # Performance metrics setup
        true_pareto_front = problem.pareto_front()
        gd_indicator = GD(true_pareto_front)
        ref_point = problem.pareto_front().max(axis=0) * 1.1
        hv_indicator = HV(ref_point)
        
        # Run optimization
        result = minimize(
            problem,
            algorithm,
            ('n_gen', n_gen),
            seed=1,
            verbose=True,
            save_history=True
        )

        # Calculate metrics
        gd_values = []
        hv_values = []
        convergence_metrics = []

        if result.history:
            for entry in result.history:
                F = entry.pop.get("F")
                gd = gd_indicator(F)
                hv = hv_indicator(F)
                gd_values.append(gd)
                hv_values.append(hv)
                convergence_metrics.append({
                    'generation': len(convergence_metrics) + 1,
                    'gd': gd,
                    'hv': hv
                })

        # Calculate execution time and additional metrics
        execution_time = time.time() - start_time
        final_gd = gd_values[-1] if gd_values else None
        final_hv = hv_values[-1] if hv_values else None
        
        performance_metrics = {
            'execution_time': execution_time,
            'final_gd': final_gd,
            'final_hv': final_hv,
            'population_size': pop_size,
            'generations': n_gen,
            'algorithm_params': data,
            'convergence_history': convergence_metrics
        }

        response = {
            'F': result.F.tolist(),
            'X': result.X.tolist(),
            'n_gen': n_gen,
            'optimal_pareto_front': problem.pareto_front().tolist(),
            'gd_history': gd_values,
            'hv_history': hv_values,
            'performance_metrics': performance_metrics,
            'success': True,
            'is_3d': problem_name in ['dtlz1', 'dtlz2'],
            'solutions': {
                'X': result.X.tolist(),
                'F': result.F.tolist(),
                'CV': result.CV.tolist() if result.CV is not None else None
            }
        }
        
        return jsonify(response)

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/export_results', methods=['POST'])
def export_results():
    try:
        data = request.json
        export_type = data.get('export_type', 'metrics')
        tab_type = data.get('tab_type', 'benchmark')
        result = data.get('result', {})

        if not result:
            return jsonify({'error': 'No optimization results found'}), 400

        if tab_type == 'benchmark':
            if export_type == 'metrics':
                # Create DataFrame for benchmark metrics with convergence history
                metrics_data = pd.DataFrame({
                    'Generation': list(range(1, len(result['gd_history']) + 1)),
                    'Generation_Distance': result['gd_history'],
                    'Hypervolume': result['hv_history'],
                    'Final_GD': result['performance_metrics']['final_gd'],
                    'Final_HV': result['performance_metrics']['final_hv'],
                    'Execution_Time': result['performance_metrics']['execution_time']
                })
                
                output = io.StringIO()
                metrics_data.to_csv(output, index=False)
                filename = 'benchmark_optimization_metrics.csv'
                
            else:  # solutions export for benchmark
                X = np.array(result['X'])
                F = np.array(result['F'])
                
                solution_data = pd.DataFrame(X, columns=[f"Decision_Variable_{i+1}" for i in range(X.shape[1])])
                for i in range(F.shape[1]):
                    solution_data[f"Objective_{i+1}"] = F[:, i]
                
                output = io.StringIO()
                solution_data.to_csv(output, index=False)
                filename = 'benchmark_optimization_solutions.csv'

        else:  # custom tab
            if export_type == 'solutions':
                X = np.array(result['X'])
                F = np.array(result['F'])
                
                solution_data = pd.DataFrame(X, columns=['x1', 'x2'])
                solution_data['Objective_1'] = F[:, 0]
                solution_data['Objective_2'] = F[:, 1]
                
                output = io.StringIO()
                solution_data.to_csv(output, index=False)
                filename = 'custom_optimization_solutions.csv'
                
            else:
                return jsonify({'error': 'Metrics export not supported for custom problems'}), 400

        # Prepare the file for download
        output.seek(0)
        return send_file(
            io.BytesIO(output.getvalue().encode('utf-8')),
            mimetype='text/csv',
            as_attachment=True,
            download_name=filename
        )

    except Exception as e:
        print(f"Export error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 400



# For Custom Functions
class ExpressionParser:
    @staticmethod
    def create_variable_mapping(expr_str):
        """Convert x[i] notation to sympy variables xi"""
        # Find all occurrences of x[number]
        pattern = r'x\[(\d+)\]'
        matches = re.finditer(pattern, expr_str)
        
        # Create mapping and replacement string
        mapping = {}
        new_expr = expr_str
        for match in matches:
            idx = match.group(1)
            var_name = f'x{idx}'
            mapping[var_name] = Symbol(var_name)
            new_expr = new_expr.replace(match.group(0), var_name)
            
        return new_expr, mapping

    @staticmethod
    def parse_expression(expr_str):
        """Parse expression string and return sympy expression and variables"""
        # Replace ^ with ** for exponentiation
        expr_str = expr_str.replace('^', '**')
        
        # Convert x[i] notation and get variable mapping
        new_expr, var_mapping = ExpressionParser.create_variable_mapping(expr_str)
        
        try:
            # Parse the expression
            expr = sympify(new_expr)
            return expr, var_mapping
        except Exception as e:
            raise ValueError(f"Invalid expression: {expr_str}. Error: {str(e)}")

class CustomProblem(ElementwiseProblem):
    def __init__(self, objectives, constraints, bounds):
        try:
            self.objective_funcs = []
            self.var_mappings = []
            vars_needed = set()

            # Parse and create objective functions
            for obj in objectives.values():
                expr, var_mapping = ExpressionParser.parse_expression(obj)
                vars_needed.update(var_mapping.keys())
                
                # Create lambda function for evaluation
                obj_func = lambdify(list(var_mapping.values()), expr)
                self.objective_funcs.append((obj_func, var_mapping))

            self.constraint_funcs = []
            if constraints:
                for constr in constraints:
                    if not all(k in constr for k in ['expression', 'operator', 'value']):
                        raise ValueError("Invalid constraint format")

                    expr, var_mapping = ExpressionParser.parse_expression(constr['expression'])
                    vars_needed.update(var_mapping.keys())
                    
                    # Create lambda function for constraint
                    constr_func = lambdify(list(var_mapping.values()), expr)
                    self.constraint_funcs.append({
                        'func': constr_func,
                        'mapping': var_mapping,
                        'operator': constr['operator'],
                        'value': float(constr['value'])
                    })

            # Validate bounds
            if not isinstance(bounds, dict) or 'xl' not in bounds or 'xu' not in bounds:
                raise ValueError("Invalid bounds format")

            xl = np.array(bounds['xl'])
            xu = np.array(bounds['xu'])

            if xl.size != xu.size:
                raise ValueError("Invalid bounds dimensions")

            if not np.all(xl < xu):
                raise ValueError("Lower bounds must be less than upper bounds")

            super().__init__(
                n_var=len(bounds['xl']),
                n_obj=len(objectives),
                n_constr=len(constraints) if constraints else 0,
                xl=xl,
                xu=xu
            )

        except Exception as e:
            raise ValueError(f"Error initializing CustomProblem: {str(e)}")

    def _evaluate(self, x, out, *args, **kwargs):
        try:
            F = []
            for obj_func, var_mapping in self.objective_funcs:
                # Create dictionary of variable values
                var_values = {f'x{i}': x[i] for i in range(len(x))}
                
                # Extract only the variables needed for this function
                func_vars = [var_values[var_name] for var_name in var_mapping.keys()]
                
                # Evaluate the function
                result = float(obj_func(*func_vars))
                F.append(result)
                
            out["F"] = F

            if self.constraint_funcs:
                G = []
                for constr in self.constraint_funcs:
                    var_values = {f'x{i}': x[i] for i in range(len(x))}
                    func_vars = [var_values[var_name] for var_name in constr['mapping'].keys()]
                    
                    val = float(constr['func'](*func_vars))
                    if constr['operator'] == '<=':
                        G.append(val - constr['value'])
                    elif constr['operator'] == '>=':
                        G.append(constr['value'] - val)
                    else:  # equality constraint
                        G.append(abs(val - constr['value']))
                out["G"] = G

        except Exception as e:
            raise ValueError(f"Error evaluating functions: {str(e)}")

def create_algorithm(name, params):
    """Helper function to create algorithm instance with parameters"""
    if name == 'NSGA2':
        return NSGA2(
            pop_size=params.get('population_size', 100),
            n_offsprings=params.get('population_size', 100)
        )
    elif name == 'NSGA3':
        ref_dirs = get_reference_directions(
            "das-dennis",
            2,  # n_obj for custom problems is always 2
            n_partitions=params.get('n_partitions', 12)
        )
        return NSGA3(
            pop_size=params.get('population_size', 100),
            ref_dirs=ref_dirs
        )
    elif name == 'MOEAD':
        ref_dirs = get_reference_directions(
            "das-dennis",
            2,  # n_obj for custom problems is always 2
            n_partitions=12
        )
        n_neighbors = min(params.get('n_neighbors', 15), len(ref_dirs) - 1)
        
        decomposition_name = params.get('decomposition', 'tchebicheff')
        if decomposition_name == 'tchebicheff':
            decomposition = Tchebicheff()
        elif decomposition_name == 'weighted_sum':
            decomposition = WeightedSum()
        elif decomposition_name == 'pbi':
            decomposition = PBI()
        else:
            raise ValueError(f'Unknown decomposition method: {decomposition_name}')
        
        return MOEAD(
            ref_dirs=ref_dirs,
            n_neighbors=n_neighbors,
            decomposition=decomposition,
            prob_neighbor_mating=params.get('prob_neighbor_mating', 0.7)
        )
    elif name == 'RNSGA2':
        ref_points = np.array([[0, 0], [1.0, 1.0]])  # Default reference points
        if 'ref_points' in params:
            ref_points = np.array(params['ref_points'])
        
        return RNSGA2(
            ref_points=ref_points,
            pop_size=params.get('population_size', 100),
            epsilon=params.get('epsilon', 0.01),
            normalization='front'
        )
    elif name == 'SPEA2':
        return SPEA2(
            pop_size=params.get('population_size', 100),
            n_offsprings=params.get('population_size', 100)
        )
    else:
        raise ValueError(f'Unknown algorithm: {name}')

@app.route('/optimize/custom', methods=['POST'])
def optimize_custom():
    try:
        data = request.json
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400

        # Validate required fields
        required_fields = ['objectives', 'bounds', 'algorithm']
        if not all(field in data for field in required_fields):
            return jsonify({'success': False, 'error': 'Missing required fields'}), 400

        # Create custom problem instance
        problem = CustomProblem(
            objectives=data['objectives'],
            constraints=data.get('constraints', []),
            bounds=data['bounds']
        )
        
        # Create algorithm instance with parameters
        algorithm = create_algorithm(data['algorithm'], data)
        
        # Run optimization
        result = minimize(
            problem,
            algorithm,
            ('n_gen', data.get('n_gen', 100)),
            seed=1,
            verbose=True
        )

        # Get the Pareto front directly from the results
        # Sort solutions by first objective for better visualization
        F = result.F
        sorted_indices = np.argsort(F[:, 0])
        sorted_F = F[sorted_indices]
        sorted_X = result.X[sorted_indices]
        
        # Calculate constraint violations if any
        CV = result.CV
        sorted_CV = CV[sorted_indices] if CV is not None else None
        
        return jsonify({
            'success': True,
            'F': sorted_F.tolist(),
            'X': sorted_X.tolist(),
            'optimal_pareto_front': sorted_F.tolist(),  # Using sorted F as the Pareto front
            'CV': sorted_CV.tolist() if sorted_CV is not None else None
        })
        
    except ValueError as ve:
        return jsonify({'success': False, 'error': str(ve)}), 400
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': f'Unexpected error: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=True)
