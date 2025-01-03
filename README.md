# Numerical Analysis Visualization

This project provides a web-based application for visualizing numerical analysis techniques such as interpolation using cubic splines and Newton's divided difference method. Additionally, it includes a Tkinter-based application for demonstrating Runge's phenomenon.

## Features

- **Web Application**:
  - Visualize interpolation of functions using cubic splines and Newton's divided difference method.
  - Supports input of functions, points, and the Runge function.
  - Displays error plots for the interpolations.
  
- **Tkinter Application**:
  - Demonstrates Runge's phenomenon.
  - Allows selection of different interpolation methods and error visualization.

## Requirements

- Python 3.x
- Flask
- Plotly
- NumPy
- SymPy
- NumExpr
- SciPy
- Tkinter (for the desktop application)

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

### Tkinter Application

1. Navigate to the project directory:
    ```sh
    cd Applied-Numerical-Bonus
    ```

2. Run the Tkinter application:
    ```sh
    python adham.py
    ```

3. Use the interface to input the number of points and select the plot option. Click "Plot" to visualize the Runge function and the selected interpolation method.

## File Structure

- `app.py`: Main Flask application file.
- `adham.py`: Tkinter application for demonstrating Runge's phenomenon.
- `templates/index.html`: HTML template for the Flask application.
- `README.md`: This README file.
