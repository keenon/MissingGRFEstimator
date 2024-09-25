import numpy as np
import matplotlib.pyplot as plt
import nimblephysics as nimble

regularization_weight = 1e2

# Define the function to generate the data
def generate_data(n_points, noise=0.1):
    x = np.linspace(0, 2*np.pi, n_points)
    y = np.sin(x) + np.random.normal(0, noise, n_points)

    dt = 1.0 / n_points
    smoothing_weight = 1.0 / (dt * dt)
    acc_minimizer = nimble.utils.AccelerationMinimizer(len(y), smoothing_weight, regularization_weight)
    output = acc_minimizer.minimize(y)

    return x, output

spacings = [50, 100, 200, 500]

# Plot the data
for spacing in spacings:
    x, y = generate_data(spacing)
    plt.plot(x, y, label=f'spacing={spacing}')

plt.legend()
plt.show()