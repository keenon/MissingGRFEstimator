import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Define the function to generate the data
def generate_data(n_points, noise=0.1):
    x = np.linspace(0, 2*np.pi, n_points)
    y = np.sin(x) + np.random.normal(0, noise, n_points)
    return x, y

# Generate the data
x, y = generate_data(100)

n = len(y)
epsilon = 0.75  # Maximum allowed deviation

# Objective function to minimize the second derivative
def objective(x):
    return np.sum((x[:-2] - 2 * x[1:-1] + x[2:]) ** 2)

# Constraints: |x_i - y_i| <= epsilon
constraints = [{'type': 'ineq', 'fun': lambda x, i=i: epsilon - abs(x[i] - y[i])} for i in range(n)]

# Initial guess (start with the noisy signal)
x0 = y.copy()

# Run the optimization
result = minimize(objective, x0, constraints=constraints, tol=1e-14)

# Optimized smooth signal
y_optimized = result.x
print("Smoothed signal:", y_optimized)

# Take two derivatives
dy = np.gradient(y_optimized, x)
d2y = np.gradient(dy, x)

# Plot the data
plt.plot(x, y, label='y')
plt.plot(x, y_optimized, label='y_optimized')
plt.plot(x, dy, label='dy')
plt.plot(x, d2y, label='d2y')
plt.legend()
plt.show()
