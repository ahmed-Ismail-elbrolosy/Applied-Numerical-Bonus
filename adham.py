import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import tkinter as tk
from tkinter import ttk, messagebox

class RungePhenomenonApp:
    def __init__(self, master):
        self.master = master
        master.title("Runge's Phenomenon")

        # Label for the number of points
        self.label = tk.Label(master, text="Number of Points:")
        self.label.pack()

        # Entry box for user input
        self.point_entry = tk.Entry(master)
        self.point_entry.pack()

        # Dropdown for plot option selection
        self.method_var = tk.StringVar(value="Cubic + Error")
        self.method_label = tk.Label(master, text="Select Plot Option:")
        self.method_label.pack()
        self.method_dropdown = ttk.Combobox(master, textvariable=self.method_var)
        self.method_dropdown['values'] = ("Cubic + Error", "Error Only", "Curve Only", "Newton + Error") 
        self.method_dropdown.pack()

        # Button to trigger the plot
        self.plot_button = tk.Button(master, text="Plot", command=self.plot)
        self.plot_button.pack()

        # Label for instructions
        self.instruction_label = tk.Label(master, text="Enter a number (e.g., 5, 10, 20):")
        self.instruction_label.pack()

    def runge_function(self, x):
        """The Runge function."""
        return 1 / (1 + 25 * x ** 2)

    def newton_interpolation(self, x, y, x_fine):
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

    def plot(self):
        """Plot the Runge function and selected interpolation method."""
        try:
            num_points = int(self.point_entry.get())
            if num_points < 2:
                raise ValueError("Number of points must be at least 2.")
        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid input: {e}")
            return

        # Generate points
        x = np.linspace(-1, 1, num_points)
        y = self.runge_function(x)

        # Create a finer grid for plotting
        x_fine = np.linspace(-1, 1, 1000)
        y_fine = self.runge_function(x_fine)

        # Cubic spline interpolation
        cs = CubicSpline(x, y)
        y_interp_cubic = cs(x_fine)

        # Newton interpolation
        y_interp_newton = self.newton_interpolation(x, y, x_fine)

        # Calculate errors
        error_cubic = np.abs(y_fine - y_interp_cubic)
        error_newton = np.abs(y_fine - y_interp_newton)
        max_error_cubic = np.max(error_cubic)
        max_error_newton = np.max(error_newton)

        # Get the selected plot option
        method = self.method_var.get()

        # Create the figures
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))  # Create 2 subplots

        if method == "Cubic + Error":
            # Plotting the Runge function and cubic spline interpolation
            axs[0].plot(x_fine, y_fine, label="Runge Function", color='blue')
            axs[0].scatter(x, y, color='red', label="Interpolation Points")
            axs[0].plot(x_fine, y_interp_cubic, label="Cubic Spline Interpolation", color='orange')
            axs[0].plot(x_fine, y_interp_newton, label="Newton Interpolation", color='purple', linestyle='-.') 
            axs[0].set_title("Cubic Spline and Newton Interpolation")
            axs[0].set_xlabel("x")
            axs[0].set_ylabel("f(x)")
            axs[0].axhline(0, color='black', lw=0.5, ls='--')
            axs[0].axvline(0, color='black', lw=0.5, ls='--')
            axs[0].legend()
            axs[0].grid()

            # Plotting the errors
            axs[1].plot(x_fine, error_cubic, label="Cubic Spline Error", color='red')
            axs[1].plot(x_fine, error_newton, label="Newton Error", color='purple')
            axs[1].set_title(f"Errors: Max Cubic = {max_error_cubic:.4e}, Max Newton = {max_error_newton:.4e}")
            axs[1].set_xlabel("x")
            axs[1].set_ylabel("Error")
            axs[1].axhline(0, color='black', lw=0.5, ls='--')
            axs[1].axvline(0, color='black', lw=0.5, ls='--')
            axs[1].legend()
            axs[1].grid()

        elif method == "Error Only":
            # Plotting the errors
            axs[0].plot(x_fine, error_cubic, label="Cubic Spline Error", color='red')
            axs[0].plot(x_fine, error_newton, label="Newton Error", color='purple')
            axs[0].set_title(f"Errors: Max Cubic = {max_error_cubic:.4e}, Max Newton = {max_error_newton:.4e}")
            axs[0].set_xlabel("x")
            axs[0].set_ylabel("Error")
            axs[0].axhline(0, color='black', lw=0.5, ls='--')
            axs[0].axvline(0, color='black', lw=0.5, ls='--')
            axs[0].legend()
            axs[0].grid()

        elif method == "Curve Only":
            # Plotting the Runge function and cubic spline interpolation
            axs[0].plot(x_fine, y_fine, label="Runge Function", color='blue')
            axs[0].scatter(x, y, color='red', label="Interpolation Points")
            axs[0].plot(x_fine, y_interp_cubic, label="Cubic Spline Interpolation", color='orange')
            axs[0].plot(x_fine, y_interp_newton, label="Newton Interpolation", color='purple', linestyle='-.') 
            axs[0].set_title("Cubic Spline and Newton Interpolation")
            axs[0].set_xlabel("x")
            axs[0].set_ylabel("f(x)")
            axs[0].axhline(0, color='black', lw=0.5, ls='--')
            axs[0].axvline(0, color='black', lw=0.5, ls='--')
            axs[0].legend()
            axs[0].grid()

        elif method == "Newton + Error":
            # Plotting the Runge function and cubic spline interpolation
            axs[0].plot(x_fine, y_fine, label="Runge Function", color='blue')
            axs[0].scatter(x, y, color='red', label="Interpolation Points")
            axs[0].plot(x_fine, y_interp_newton, label="Newton Interpolation", color='purple', linestyle='-.')
            axs[0].set_title("Newton Interpolation")
            axs[0].set_xlabel("x")
            axs[0].set_ylabel("f(x)")
            axs[0].axhline(0, color='black', lw=0.5, ls='--')
            axs[0].axvline(0, color='black', lw=0.5, ls='--')
            axs[0].legend()
            axs[0].grid()

            # Plotting the errors
            axs[1].plot(x_fine, error_newton, label="Newton Error", color='purple')
            axs[1].set_title(f"Errors: Max Newton = {max_error_newton:.4e}")
            axs[1].set_xlabel("x")
            axs[1].set_ylabel("Error")
            axs[1].axhline(0, color='black', lw=0.5, ls='--')
            axs[1].axvline(0, color='black', lw=0.5, ls='--')
            axs[1].legend()
            axs[1].grid()
        # Show the plots
        plt.tight_layout()  # Adjust spacing between plots
        plt.show()

if __name__ == "__main__":
    root = tk.Tk()
    app = RungePhenomenonApp(root)
    root.mainloop()