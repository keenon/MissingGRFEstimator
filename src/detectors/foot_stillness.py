import nimblephysics as nimble
from typing import List
from detectors.abstract_detector import AbstractDetector


class FootStillnessDetector(AbstractDetector):
    def __init__(self):
        pass

    def estimate_missing_grfs(self, subject: nimble.biomechanics.SubjectOnDisk, trials: List[int]) -> List[List[bool]]:
        pass
