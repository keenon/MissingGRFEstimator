import time

from cli.abstract_command import AbstractCommand
import os
import nimblephysics as nimble
import numpy as np
from typing import List, Tuple
from scipy.signal import butter, filtfilt


class IMUAnalyze6(AbstractCommand):
    def __init__(self):
        pass

    def register_subcommand(self, subparsers):
        subparser = subparsers.add_parser('imu-analyze6', help='Compare the accelerations of the IMUs.')
        subparser.add_argument('--dataset-home', type=str, default='../data', help='The path to the AddBiomechanics dataset.')

    def run(self, args):
        if 'command' in args and args.command != 'imu-analyze6':
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

        skel = subject_on_disk.readSkel(subject_on_disk.getNumProcessingPasses()-1)

        accelerometers: List[Tuple[nimble.dynamics.BodyNode, nimble.math.Isometry3]] = []
        foot_translation = np.array([0.17, 0.07, 0.0])
        accelerometers.append((skel.getBodyNode('calcn_l'), nimble.math.Isometry3(np.eye(3), foot_translation)))
        tibia_translation = np.array([0.05, -0.2, 0.0])
        accelerometers.append((skel.getBodyNode('tibia_l'), nimble.math.Isometry3(np.eye(3), tibia_translation)))
        femur_translation = np.array([0.05, -0.2, 0.0])
        accelerometers.append((skel.getBodyNode('femur_l'), nimble.math.Isometry3(np.eye(3), femur_translation)))
        torso_translation = np.array([0.1, 0.3, 0.0])
        accelerometers.append((skel.getBodyNode('torso'), nimble.math.Isometry3(np.eye(3), torso_translation)))

        for trial, path in trial_pairs:
            trial_len = subject_on_disk.getTrialLength(trial)
            dt = subject_on_disk.getTrialTimestep(trial)

            # First lowpass butterworth filter the data with scipy
            fs = 1.0 / dt
            cutoff = 10.0
            b, a = butter(4, cutoff / (0.5 * fs), btype='low', analog=False)

            # Get the data from the CSV file
            data = np.genfromtxt(path, delimiter=',', skip_header=1)
            csv_time = data[:, 0]

            # Organize the IMU time series by IMU
            raw_imu_data = [
                data[:, 1:4],  # foot
                data[:, 7:10],  # shank
                data[:, 13:16],  # thigh
                data[:, 19:22]  # torso
            ]
            imu_names = ['foot', 'shank', 'thigh', 'torso']

            # Get the data from the B3D file
            trial_len = subject_on_disk.getTrialLength(trial)
            frames = subject_on_disk.readFrames(trial, 0, trial_len)
            num_dofs = skel.getNumDofs()
            raw_poses = np.zeros((num_dofs, trial_len))
            for t in range(trial_len):
                frame = frames[t]
                raw_poses[:, t] = frame.processingPasses[0].pos
                if t > 0:
                    raw_poses[:, t] = skel.unwrapPositionToNearest(raw_poses[:, t], raw_poses[:, t-1])

            # for dof in range(num_dofs):
            #     raw_poses[dof, :] = filtfilt(b, a, raw_poses[dof, :])

            pose_regularization = 1000.0

            acceleration_minimizer = nimble.utils.AccelerationMinimizer(trial_len, 1.0 / (dt * dt), pose_regularization)
            positions = np.zeros_like(raw_poses)
            for dof in range(num_dofs):
                if pose_regularization > 0:
                    positions[dof, :] = acceleration_minimizer.minimize(raw_poses[dof, :])
                else:
                    positions[dof, :] = raw_poses[dof, :]

            velocities = np.zeros((num_dofs, trial_len))
            for t in range(1, trial_len):
                velocities[:, t] = (positions[:, t] - positions[:, t-1]) / dt
            if trial_len > 1:
                velocities[:, 0] = velocities[:, 1]

            accelerations = np.zeros((num_dofs, trial_len))
            for t in range(1, trial_len):
                accelerations[:, t] = (velocities[:, t] - velocities[:, t-1]) / dt
            if trial_len > 1:
                accelerations[:, 0] = accelerations[:, 1]

            imu_readings = [np.zeros((trial_len, 3))] * len(accelerometers)
            for t in range(trial_len):
                skel.setPositions(positions[:, t])
                skel.setVelocities(velocities[:, t])
                skel.setAccelerations(accelerations[:, t])
                imu_reading = skel.getAccelerometerReadings(accelerometers)
                for i in range(len(accelerometers)):
                    imu_readings[i][t, :] = imu_reading[i*3:i*3+3]

            b3d_times = np.array(list(range(trial_len))) * subject_on_disk.getTrialTimestep(trial)
            csv_time -= csv_time[0]

            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(4, 1)
            fig.suptitle('IMU Accelerations')

            for i in range(4):
                min_length = min(len(b3d_times), len(csv_time))
                axs[i].plot(b3d_times[:min_length], np.linalg.norm(imu_readings[i], axis=1)[:min_length], label=imu_names[i]+ ' (synthetic)')
                axs[i].plot(csv_time[:min_length], np.linalg.norm(raw_imu_data[i], axis=1)[:min_length] * 9.81, label=imu_names[i] + ' (measured)')
                axs[i].set_title(imu_names[i])
                axs[i].set(xlabel='Time (s)', ylabel='Acceleration (m/s^2)')
                axs[i].legend()

            plt.show()
