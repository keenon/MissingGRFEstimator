import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Define the function to generate the data
def generate_data(n_points, noise=0.01):
    x = np.linspace(0, 2*np.pi, n_points)
    y = np.sin(x) + np.random.normal(0, noise, n_points)
    return x, y

# Generate the data
x, y = generate_data(100)

# Inject a random outlier
y[29] += 0.25
y[30] += 0.5
y[31] += 0.25

n = len(y)
epsilon = 0.75  # Maximum allowed deviation

# Objective function to minimize the second derivative
def objective(x):
    acc_loss = np.sum((x[:-2] - 2 * x[1:-1] + x[2:]) ** 2)
    original_loss = np.sum((x - y) ** 2)
    # original_loss = np.sum(np.abs(x - y))
    return acc_loss + 0.05 * original_loss

# Initial guess (start with the noisy signal)
x0 = y.copy()

# Run the optimization
result = minimize(objective, x0, tol=1e-5)

print(result)

# Optimized smooth signal
y_optimized = result.x
print("Smoothed signal:", y_optimized)

# Objective function to minimize the second derivative
def refined_objective(x):
    acc_loss = np.sum((x[:-2] - 2 * x[1:-1] + x[2:]) ** 2)
    original_loss = np.sum((x - y) ** 2)
    original_loss = np.sum(np.sqrt(np.abs(x - y)))
    return acc_loss + 0.00001 * original_loss

result2 = minimize(refined_objective, y_optimized, tol=1e-5, method='Powell')

print(result2)

# Optimized smooth signal
y_optimized = y
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
