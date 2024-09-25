from cli.abstract_command import AbstractCommand
import os
import nimblephysics as nimble
import numpy as np
from typing import List, Tuple


class IMUAnalyze2(AbstractCommand):
    def __init__(self):
        pass

    def register_subcommand(self, subparsers):
        subparser = subparsers.add_parser('imu-analyze2', help='Analyze the raw IMU data.')
        subparser.add_argument('--dataset-home', type=str, default='../data', help='The path to the AddBiomechanics dataset.')

    def run(self, args):
        if 'command' in args and args.command != 'imu-analyze2':
            return False
        dataset_home: str = args.dataset_home
        print(f"Running evaluation on {dataset_home}")

        data_dir = os.path.abspath('../data/smoothed/protected/us-west-2:e013a4d2-683d-48b9-bfe5-83a0305caf87/data/Camargo2021_Formatted_No_Arm/AB18_split4')

        # Recursively list all the B3D files under data
        b3d_files = []
        csv_files = []
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith(".b3d"):
                    b3d_files.append(os.path.join(root, file))
                if file.endswith(".csv"):
                    csv_files.append(os.path.join(root, file))

        assert(len(b3d_files) == 1)
        subject_on_disk = nimble.biomechanics.SubjectOnDisk(b3d_files[0])

        desired_trial_name = 'stair_3_r_01_02_segment_0'
        desired_trial_index = -1
        for trial in range(subject_on_disk.getNumTrials()):
            if subject_on_disk.getTrialName(trial) == desired_trial_name:
                print(f"Found trial {desired_trial_name} at index {trial}")
                desired_trial_index = trial
                break
        assert desired_trial_index != -1

        trial = desired_trial_index
        skel = subject_on_disk.readSkel(subject_on_disk.getNumProcessingPasses()-1, ignoreGeometry=True)
        accelerometers: List[Tuple[nimble.dynamics.BodyNode, nimble.math.Isometry3]] = []
        # accelerometers.append((skel.getBodyNode('calcn_r'), nimble.math.Isometry3()))
        accelerometers.append((skel.getBodyNode('pelvis'), nimble.math.Isometry3()))

        trial_len = subject_on_disk.getTrialLength(trial)
        frames = subject_on_disk.readFrames(trial, 0, trial_len)
        imu_readings = np.zeros((trial_len, 3))
        for t in range(trial_len):
            frame = frames[t]
            dynamics_pass = frame.processingPasses[-1]
            skel.setPositions(dynamics_pass.pos)
            skel.setVelocities(dynamics_pass.vel)
            skel.setAccelerations(dynamics_pass.acc)
            imu_reading = skel.getAccelerometerReadings(accelerometers)
            imu_readings[t] = imu_reading

        # Plot the data
        import matplotlib.pyplot as plt
        time = np.arange(trial_len) / 100
        plt.plot(time, imu_readings[:, 0], label='pelvis X')
        plt.plot(time, imu_readings[:, 1], label='pelvis Y')
        plt.plot(time, imu_readings[:, 2], label='pelvis Z')
        plt.legend()
        plt.title(desired_trial_name)
        plt.show()