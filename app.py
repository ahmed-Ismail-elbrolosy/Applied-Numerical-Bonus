from flask import Flask, render_template, request
import plotly.graph_objs as go
import plotly.io as pio
import numpy as np
import sympy as sp
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
    plot_div = ""
    try:
        if request.method == 'POST':
            x0 = float(request.form['x0'])
            y0 = float(request.form['y0'])
            step_value = int(request.form['step_value'])
            equation = request.form['equation']
            algorithms = request.form.getlist('algorithms')

            x = np.linspace(x0, y0, step_value)
            x_sym = sp.symbols('x')
            y_exact = [float(sp.sympify(equation).subs(x_sym, val)) for val in x]

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=y_exact, mode='lines', name='Exact Equation', line=dict(color='black', width=1)))

            if 'cubic_spline' in algorithms:
                cs = CubicSpline(x, y_exact)
                y_interp_cubic = cs(x)
                fig.add_trace(go.Scatter(x=x, y=y_interp_cubic, mode='lines', name='Cubic Spline', line=dict(color='orange', width=2)))

            if 'newton' in algorithms:
                y_interp_newton = newton_interpolation(x, y_exact, x)
                fig.add_trace(go.Scatter(x=x, y=y_interp_newton, mode='lines', name='Newton', line=dict(color='purple', width=2)))

            if 'vandermonde' in algorithms:
                y_interp_vander = vandermonde_interpolation(x, y_exact, x)
                fig.add_trace(go.Scatter(x=x, y=y_interp_vander, mode='lines', name='Vandermonde', line=dict(color='green', width=2)))

            plot_div = pio.to_html(fig, full_html=False)
    except Exception as e:
        logging.error("Error occurred", exc_info=True)
        plot_div = f"<div class='alert alert-danger'>An error occurred: {e}</div>"

    return render_template('index.html', plot_div=plot_div)

if __name__ == '__main__':
    app.run(debug=True)
