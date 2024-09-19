import os
import nimblephysics as nimble
import numpy as np

path = '../../data/raw_wang_subject_12_markers'

# Recursively list all the TRC files under data
trc_files = []
for root, dirs, files in os.walk(os.path.abspath(path)):
    for file in files:
        if file.endswith(".trc"):
            trc_files.append(os.path.join(root, file))

# Load all the marker data
magnitudes = []
for file in trc_files:
    trc = nimble.biomechanics.OpenSimParser.loadTRC(file)
    marker_lines = trc.markerLines
    for key in marker_lines:
        marker_line = marker_lines[key]
        for i in range(len(marker_line)):
            marker = marker_line[i]
            magnitude = np.linalg.norm(marker)
            magnitudes.append(magnitude)

# Plot the histogram of marker magnitudes
import matplotlib.pyplot as plt
plt.hist(magnitudes, bins=100)
plt.xlabel('Magnitude')
plt.ylabel('Count')

plt.show()