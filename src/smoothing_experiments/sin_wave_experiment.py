import numpy as np
import matplotlib.pyplot as plt

# Define the function to generate the data
def generate_data(n_points, noise=0.1):
    x = np.linspace(0, 2*np.pi, n_points)
    y = np.sin(x) + np.random.normal(0, noise, n_points)
    return x, y

# Generate the data
x, y = generate_data(100)

# Take two derivatives
dy = np.gradient(y, x)
d2y = np.gradient(dy, x)

# FFT the signal
fft = np.fft.fft(y)
fft_dy = np.fft.fft(dy)
fft_d2y = np.fft.fft(d2y)
fft_freq = np.fft.fftfreq(len(y), x[1] - x[0])
fft_freq_dy = np.fft.fftfreq(len(dy), x[1] - x[0])
fft_freq_d2y = np.fft.fftfreq(len(d2y), x[1] - x[0])

# Apply the low-pass filter by zeroing frequencies above the cutoff
fft_filtered = fft.copy()
fft_filtered[np.abs(fft_freq) > 1.0] = 0

# Inverse FFT to transform back to the time domain
filtered_signal = np.fft.ifft(fft_filtered).real
y = filtered_signal

# Plot the data
plt.plot(x, y, label='y')
plt.show()

# Plot the data
plt.plot(fft_freq, np.abs(fft), label='y')
plt.plot(fft_freq_dy, np.abs(fft_dy), label='dy')
plt.plot(fft_freq_d2y, np.abs(fft_d2y), label='d2y')
plt.legend()
plt.show()

# Plot the data
plt.plot(x, y, label='y')
plt.plot(x, dy, label='dy')
plt.plot(x, d2y, label='d2y')
plt.legend()
plt.show()