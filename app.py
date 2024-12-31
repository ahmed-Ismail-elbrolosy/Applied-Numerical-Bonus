from flask import Flask, render_template, request, jsonify
import plotly.graph_objs as go
import plotly.io as pio
import numpy as np
import sympy as sp
import numexpr as ne
from scipy.interpolate import CubicSpline
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

def runge_function(x):
    """The Runge function."""
    return 1 / (1 + 25 * x ** 2)

def vandermonde_interpolation(x, y, x_fine):
    """Perform Vandermonde interpolation."""
    A = np.vander(x, increasing=True)
    coeffs = np.linalg.solve(A, y)
    return np.polyval(coeffs[::-1], x_fine)

def newton_interpolation(x, y, x_fine):
    """Perform Newton's divided difference interpolation."""
    n = len(x)
    divided_diff = np.zeros((n, n))
    divided_diff[:, 0] = y

    # Calculate divided differences
    for j in range(1, n):
        for i in range(n - j):
            divided_diff[i, j] = (divided_diff[i + 1, j - 1] - divided_diff[i, j - 1]) / (x[i + j] - x[i])

    # Evaluate the polynomial
    y_interp_newton = y[0]
    for j in range(1, n):
        term = divided_diff[0, j]
        for k in range(j):
            term *= (x_fine - x[k])
        y_interp_newton += term

    return y_interp_newton

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        logging.debug("Received POST request")
        try:
            x0 = request.form.get('x0', type=float)
            y0 = request.form.get('y0', type=float)
            step_value = request.form.get('step_value', type=int)
            equation = request.form.get('equation', '')
            algorithms = request.form.getlist('algorithms')
            x_values = request.form.getlist('x_values[]')
            y_values = request.form.getlist('y_values[]')

            logging.debug(f"x0: {x0}, y0: {y0}, step_value: {step_value}, equation: {equation}, algorithms: {algorithms}")

            if x0 is not None and y0 is not None and step_value is not None:
                x = np.linspace(x0, y0, step_value)
                x_fine = np.linspace(x0, y0)  # For a continuous plot
            else:
                x = np.array([])
                x_fine = np.array([])

            if x_values and y_values and all(x_values) and all(y_values):
                x = np.array([float(x) for x in x_values if x])
                y_exact_points = np.array([float(y) for y in y_values if y])
                sorted_indices = np.argsort(x)
                x = x[sorted_indices]
                y_exact_points = y_exact_points[sorted_indices]
            else:
                if not equation:
                    raise ValueError("Equation is required if no points are provided.")
                x_sym = sp.symbols('x')
                equation = equation.replace('^', '**')
                y_exact_points = ne.evaluate(equation, local_dict={'x': x})

            fig = go.Figure()

            if equation:
                y_exact = ne.evaluate(equation, local_dict={'x': x_fine})
                fig.add_trace(go.Scatter(x=x_fine, y=y_exact, mode='lines', name='Exact Equation', line=dict(color='black', width=1)))

            fig.add_trace(go.Scatter(x=x, y=y_exact_points, mode='markers', name='Points', marker=dict(color='red', size=4)))

            points = [{'x': float(x_val), 'y': float(y_val)} for x_val, y_val in zip(x, y_exact_points)]

            if 'cubic_spline' in algorithms:
                cs = CubicSpline(x, y_exact_points)
                y_interp_cubic = cs(x_fine if x_fine.size > 0 else np.linspace(x.min(), x.max(), 1000))
                fig.add_trace(go.Scatter(x=x_fine if x_fine.size > 0 else np.linspace(x.min(), x.max(), 1000), y=y_interp_cubic, mode='lines', name='Cubic Spline', line=dict(color='orange', width=2)))
                y_interp_cubic_points = cs(x)
                for point, y_val in zip(points, y_interp_cubic_points):
                    point['cubic_spline'] = float(y_val)
                    # Calculate the cubic spline error using its own formula
                    point['cubic_spline_error'] = abs(cs(point['x']) - point['y'])

            if 'newton' in algorithms:
                y_interp_newton = newton_interpolation(x, y_exact_points, x_fine if x_fine.size > 0 else np.linspace(x.min(), x.max(), 1000))
                fig.add_trace(go.Scatter(x=x_fine if x_fine.size > 0 else np.linspace(x.min(), x.max(), 1000), y=y_interp_newton, mode='lines', name='Newton', line=dict(color='purple', width=2)))
                y_interp_newton_points = newton_interpolation(x, y_exact_points, x)
                for point, y_val in zip(points, y_interp_newton_points):
                    point['newton'] = float(y_val)
                    point['newton_error'] = abs(point['y'] - point['newton'])

            if 'vandermonde' in algorithms:
                y_interp_vander = vandermonde_interpolation(x, y_exact_points, x_fine if x_fine.size > 0 else np.linspace(x.min(), x.max(), 1000))
                fig.add_trace(go.Scatter(x=x_fine if x_fine.size > 0 else np.linspace(x.min(), x.max(), 1000), y=y_interp_vander, mode='lines', name='Vandermonde', line=dict(color='green', width=2)))
                y_interp_vander_points = vandermonde_interpolation(x, y_exact_points, x)
                for point, y_val in zip(points, y_interp_vander_points):
                    point['vandermonde'] = float(y_val)
                    point['vandermonde_error'] = abs(point['y'] - point['vandermonde'])

            plot_html = pio.to_html(fig, full_html=False)
            return jsonify({'plot_html': plot_html, 'points': points})
        except Exception as e:
            logging.error("Error occurred", exc_info=True)
            return str(e), 400
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
