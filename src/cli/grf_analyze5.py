import time
import random
from cli.abstract_command import AbstractCommand
import os
import nimblephysics as nimble
import numpy as np
import traceback
from typing import List, Tuple


class GRFAnalyze5(AbstractCommand):
    def __init__(self):
        pass

    def register_subcommand(self, subparsers):
        subparser = subparsers.add_parser('grf-analyze5', help='Examine why certain trials have such dramatic trajectory shifts when we try to match the acceleration too closely.')
        subparser.add_argument('--dataset-home', type=str, default='../data', help='The path to the AddBiomechanics dataset.')

    def run(self, args):
        if 'command' in args and args.command != 'grf-analyze5':
            return False
        dataset_home: str = args.dataset_home
        print(f"Running evaluation on {dataset_home}")

        file_trial_pairs = [
            ("/Users/keenonwerling/Desktop/dev/MissingGRFEstimator/data/smoothed/protected/us-west-2:e013a4d2-683d-48b9-bfe5-83a0305caf87/data/Carter2023_Formatted_No_Arm/P011_split0/P011_split0.b3d", 25),
            ("/Users/keenonwerling/Desktop/dev/MissingGRFEstimator/data/smoothed/protected/us-west-2:e013a4d2-683d-48b9-bfe5-83a0305caf87/data/Camargo2021_Formatted_No_Arm/AB23_split5/AB23_split5.b3d", 1)
        ]

        for file, trial in file_trial_pairs:
            print("Loading: " + file)
            try:
                subject: nimble.biomechanics.SubjectOnDisk = nimble.biomechanics.SubjectOnDisk(file)
                subject.loadAllFrames(doNotStandardizeForcePlateData=True)

                trial_protos = subject.getHeaderProto().getTrials()

                trial_len = subject.getTrialLength(trial)
                dt = subject.getTrialTimestep(trial)
                pose_regularization = 1000.0
                missing_grf = subject.getMissingGRF(trial)
                trial_proto = trial_protos[trial]

                # Copy force plate data to Python
                raw_force_plates = trial_proto.getForcePlates()
                force_plate_raw_forces: List[List[np.ndarray]] = [force_plate.forces for force_plate in raw_force_plates]
                force_plate_raw_cops: List[List[np.ndarray]] = [force_plate.centersOfPressure for force_plate in raw_force_plates]
                force_plate_raw_moments: List[List[np.ndarray]] = [force_plate.moments for force_plate in raw_force_plates]

                force_plate_norms: List[np.ndarray] = [np.zeros(trial_len) for _ in
                                                       range(len(raw_force_plates))]
                for i in range(len(raw_force_plates)):
                    force_norms = force_plate_norms[i]
                    for t in range(trial_len):
                        force_norms[t] = np.linalg.norm(force_plate_raw_forces[i][t])
                # 4. Next, low-pass filter the GRF data for each non-zero section
                lowpass_force_plates: List[nimble.biomechanics.ForcePlate] = []
                for i in range(len(raw_force_plates)):
                    force_matrix = np.zeros((3, trial_len))
                    cop_matrix = np.zeros((3, trial_len))
                    moment_matrix = np.zeros((3, trial_len))
                    force_norms = force_plate_norms[i]
                    non_zero_segments: List[Tuple[int, int]] = []
                    last_nonzero = -1
                    # 4.1. Find the non-zero segments
                    for t in range(trial_len):
                        if force_norms[t] > 0.0:
                            if last_nonzero < 0:
                                last_nonzero = t
                        else:
                            if last_nonzero >= 0:
                                non_zero_segments.append((last_nonzero, t))
                                last_nonzero = -1
                        force_matrix[:, t] = force_plate_raw_forces[i][t]
                        cop_matrix[:, t] = force_plate_raw_cops[i][t]
                        moment_matrix[:, t] = force_plate_raw_moments[i][t]
                    if last_nonzero >= 0:
                        non_zero_segments.append((last_nonzero, trial_len))

                    # 4.2. Lowpass filter each non-zero segment
                    for start, end in non_zero_segments:
                        # print(f"Filtering force plate {i} on non-zero range [{start}, {end}]")
                        if end - start < 10:
                            # print(" - Skipping non-zero segment because it's too short. Zeroing instead")
                            for t in range(start, end):
                                force_plate_raw_forces[i][t] = np.zeros(3)
                                force_plate_raw_cops[i][t] = np.zeros(3)
                                force_plate_raw_moments[i][t] = np.zeros(3)
                                force_norms[t] = 0.0
                        else:
                            start_weight = 1e5 if start > 0 else 0.0
                            end_weight = 1e5 if end < trial_len else 0.0
                            input_force_dim = end - start
                            input_force_start_index = 0
                            input_force_end_index = input_force_dim
                            if start_weight > 0:
                                input_force_dim += 5
                                input_force_start_index += 5
                                input_force_end_index += 5
                            if end_weight > 0:
                                input_force_dim += 5
                            padded_start = start
                            if start_weight > 0:
                                padded_start -= 5
                            padded_end = end
                            if end_weight > 0:
                                padded_end += 5

                            acc_minimizer = nimble.utils.AccelerationMinimizer(input_force_dim,
                                                                               1.0 / (dt * dt),
                                                                               pose_regularization,
                                                                               startPositionZeroWeight=start_weight,
                                                                               endPositionZeroWeight=end_weight,
                                                                               startVelocityZeroWeight=start_weight,
                                                                               endVelocityZeroWeight=end_weight)
                            cop_acc_minimizer = nimble.utils.AccelerationMinimizer(end - start,
                                                                                   1.0 / (dt * dt),
                                                                                   pose_regularization)

                            for j in range(3):
                                input_force = np.zeros(input_force_dim)
                                input_force[input_force_start_index:input_force_end_index] = force_matrix[j, start:end]
                                input_moment = np.zeros(input_force_dim)
                                input_moment[input_force_start_index:input_force_end_index] = moment_matrix[j, start:end]

                                smoothed_force = acc_minimizer.minimize(input_force)
                                if np.sum(np.abs(smoothed_force)) != 0:
                                    smoothed_force *= np.sum(np.abs(force_matrix[j, start:end])) / np.sum(np.abs(smoothed_force))

                                # average_force_distance = np.mean(np.abs(smoothed_force - input_force))
                                # print(f" - Average force distance: {average_force_distance}")
                                # if average_force_distance > 0.05:
                                #     print(f" - Force distance too high, plotting")
                                #     print("File: " + file)
                                #     print("Trial: " + str(trial))
                                #     print("Start: " + str(start))
                                #     print("End: " + str(end))
                                #     import matplotlib.pyplot as plt
                                #     plt.plot(input_force, label='Input')
                                #     plt.plot(smoothed_force, label='Output')
                                #     plt.legend()
                                #     plt.show()
                                force_matrix[j, padded_start:padded_end] = smoothed_force

                                smoothed_moment = acc_minimizer.minimize(input_moment)
                                if np.sum(np.abs(smoothed_moment)) != 0:
                                    smoothed_moment *= np.sum(np.abs(moment_matrix[j, start:end])) / np.sum(np.abs(smoothed_moment))
                                moment_matrix[j, padded_start:padded_end] = smoothed_moment

                                # We don't restrict the CoP dynamics at the beginning or end of a stride, so we don't
                                # need to pad the input to account for ramping up or down to zero.
                                cop_matrix[j, start:end] = cop_acc_minimizer.minimize(cop_matrix[j, start:end])

                            for t in range(start, end):
                                force_plate_raw_forces[i][t] = force_matrix[:, t]
                                force_plate_raw_cops[i][t] = cop_matrix[:, t]
                                force_plate_raw_moments[i][t] = moment_matrix[:, t]

                    # 4.3. Create a new lowpass filtered force plate
                    force_plate_copy = nimble.biomechanics.ForcePlate.copyForcePlate(raw_force_plates[i])
                    force_plate_copy.forces = force_plate_raw_forces[i]
                    force_plate_copy.centersOfPressure = force_plate_raw_cops[i]
                    force_plate_copy.moments = force_plate_raw_moments[i]
                    lowpass_force_plates.append(force_plate_copy)

                # print("Output COM accs: " + str(np.mean(output_com_acc, axis=1)))

            except Exception as e:
                print("Error loading: " + file)
                print(e)
                traceback.print_exc()