# Numerical Analysis Visualization

This project provides a web-based application for visualizing numerical analysis techniques such as interpolation using cubic splines and Newton's divided difference method.

## Features

- **Web Application**:
  - Visualize interpolation of functions using cubic splines and Newton's divided difference method.
  - Supports input of functions, points, and the Runge function.
  - Displays error plots for the interpolations.

## Requirements

- Python 3.x
- Flask
- Plotly
- NumPy
- SymPy
- NumExpr
- SciPy

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/ahmed-Ismail-elbrolosy/Applied-Numerical-Bonus
    cd Applied-Numerical-Bonus
    ```

2. Install the required Python packages:
    ```sh
    pip install flask plotly numpy sympy numexpr scipy
    ```

## Usage

### Web Application

1. Navigate to the project directory:
    ```sh
    cd Applied-Numerical-Bonus
    ```

2. Run the Flask application:
    ```sh
    python app.py
    ```

3. Open your web browser and go to `http://127.0.0.1:5000/`.

4. Use the interface to select the mode (Equation, Points, or Runge Function), input the required parameters, and choose the interpolation algorithms. Click "Generate" to visualize the results.

## File Structure

- `app.py`: Main Flask application file.
- `templates/index.html`: HTML template for the Flask application.
- `README.md`: This README file.
