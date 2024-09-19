import nimblephysics as nimble
import numpy as np

file = "/Users/keenonwerling/Desktop/dev/MissingGRFEstimator/data/processed/protected/us-west-2:e013a4d2-683d-48b9-bfe5-83a0305caf87/data/Wang2023_Formatted_No_Arm/Subj12/Subj12.b3d"
# trial = 1

max_magnitude = 0.0
min_magnitude = 1000000.0

subject = nimble.biomechanics.SubjectOnDisk(file)
for trial in range(subject.getNumTrials()):
    num_frames = subject.getTrialLength(trial)
    frames = subject.readFrames(trial, 0, num_frames, includeSensorData=True, includeProcessingPasses=False)
    for i in range(num_frames):
        frame = frames[i]
        markers = frame.markerObservations
        for marker_name, marker_pos in markers:
            magnitude = np.linalg.norm(marker_pos)
            if magnitude > max_magnitude:
                max_trial = trial
                max_magnitude = magnitude
                max_marker = marker_name
                max_pos = marker_pos
            if magnitude < min_magnitude:
                min_trial = trial
                min_magnitude = magnitude
                min_marker = marker_name
                min_pos = marker_pos

print(max_trial, max_marker, max_pos, max_magnitude)
print(min_trial, min_marker, min_pos, min_magnitude)