from cli.abstract_command import AbstractCommand
import os
import nimblephysics as nimble
import numpy as np
from typing import List, Tuple


class IMUAnalyze3(AbstractCommand):
    def __init__(self):
        pass

    def register_subcommand(self, subparsers):
        subparser = subparsers.add_parser('imu-analyze3', help='Analyze the raw IMU data.')
        subparser.add_argument('--dataset-home', type=str, default='../data', help='The path to the AddBiomechanics dataset.')

    def run(self, args):
        if 'command' in args and args.command != 'imu-analyze3':
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

        trial_pairs: List[Tuple[int, str]] = []
        for trial in range(subject_on_disk.getNumTrials()):
            if subject_on_disk.getTrialName(trial).endswith('_segment_0'):
                original_trial_name = subject_on_disk.getTrialName(trial)[0:-len('_segment_0')]
                filtered_csv_files = [csv_file for csv_file in csv_files if original_trial_name in csv_file]
                if len(filtered_csv_files) == 1:
                    trial_pairs.append((trial, filtered_csv_files[0]))
                break

        skel = subject_on_disk.readSkel(subject_on_disk.getNumProcessingPasses()-1, ignoreGeometry=True)
        accelerometers: List[Tuple[nimble.dynamics.BodyNode, nimble.math.Isometry3]] = []
        # accelerometers.append((skel.getBodyNode('calcn_r'), nimble.math.Isometry3()))
        accelerometers.append((skel.getBodyNode('torso'), nimble.math.Isometry3()))

        for trial, path in trial_pairs:
            # Get the data from the CSV file
            data = np.genfromtxt(path, delimiter=',', skip_header=1)
            csv_time = data[:, 0]
            csv_pelvis_x = data[:, 19]
            csv_pelvis_y = data[:, 20]
            csv_pelvis_z = data[:, 21]
            # Convert to m/s^2
            csv_magnitudes = np.sqrt(csv_pelvis_x**2 + csv_pelvis_y**2 + csv_pelvis_z**2) * 9.81

            # Get the data from the B3D file
            trial_len = subject_on_disk.getTrialLength(trial)
            frames = subject_on_disk.readFrames(trial, 0, trial_len)
            b3d_times = np.array(list(range(trial_len))) * subject_on_disk.getTrialTimestep(trial)
            imu_readings = np.zeros((trial_len, 3))
            for t in range(trial_len):
                frame = frames[t]
                dynamics_pass = frame.processingPasses[1]
                skel.setPositions(dynamics_pass.pos)
                skel.setVelocities(dynamics_pass.vel)
                skel.setAccelerations(dynamics_pass.acc)
                imu_reading = skel.getAccelerometerReadings(accelerometers)
                imu_readings[t] = imu_reading
            b3d_magnitudes = np.linalg.norm(imu_readings, axis=1)

            # Plot the data
            import matplotlib.pyplot as plt
            csv_time -= csv_time[0]
            b3d_times -= b3d_times[0]
            plt.plot(csv_time, csv_magnitudes, label='CSV')
            plt.plot(b3d_times, b3d_magnitudes, label='B3D')
            plt.legend()
            plt.title(subject_on_disk.getTrialName(trial))
            plt.show()