import nimblephysics as nimble
from typing import List
from detectors.abstract_detector import AbstractDetector
import numpy as np


class ZeroGRFDetector(AbstractDetector):
    def __init__(self):
        pass

    def estimate_missing_grfs(self, subject: nimble.biomechanics.SubjectOnDisk, trials: List[int]) -> List[List[bool]]:
        result: List[List[bool]] = []
        for trial in trials:
            trial_len = subject.getTrialLength(trial)
            missing = []
            for i in range(trial_len):
                frame = subject.readFrames(trial, i, 1, includeSensorData=True, includeProcessingPasses=False)[0]
                forces = frame.rawForcePlateForces
                total_force_mag = 0.0
                for force in forces:
                    total_force_mag += np.linalg.norm(force)
                missing.append(total_force_mag < 10.0)
            result.append(missing)
        return result
