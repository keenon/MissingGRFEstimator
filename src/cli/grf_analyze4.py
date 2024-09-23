import time
import random
from cli.abstract_command import AbstractCommand
import os
import nimblephysics as nimble
import numpy as np
import traceback


class GRFAnalyze4(AbstractCommand):
    def __init__(self):
        pass

    def register_subcommand(self, subparsers):
        subparser = subparsers.add_parser('grf-analyze4', help='Examine why certain trials have such dramatic trajectory shifts when we try to match the acceleration too closely.')
        subparser.add_argument('--dataset-home', type=str, default='../data', help='The path to the AddBiomechanics dataset.')

    def run(self, args):
        if 'command' in args and args.command != 'grf-analyze4':
            return False
        dataset_home: str = args.dataset_home
        print(f"Running evaluation on {dataset_home}")

        file_trial_pairs = [
            ('/Users/keenonwerling/Desktop/dev/MissingGRFEstimator/data/smoothed/protected/us-west-2:e013a4d2-683d-48b9-bfe5-83a0305caf87/data/Wang2023_Formatted_No_Arm/Subj04/Subj04.b3d', 20),
            ("/Users/keenonwerling/Desktop/dev/MissingGRFEstimator/data/smoothed/protected/us-west-2:e013a4d2-683d-48b9-bfe5-83a0305caf87/data/Carter2023_Formatted_No_Arm/P011_split0/P011_split0.b3d", 25),
            ("/Users/keenonwerling/Desktop/dev/MissingGRFEstimator/data/smoothed/protected/us-west-2:e013a4d2-683d-48b9-bfe5-83a0305caf87/data/Camargo2021_Formatted_No_Arm/AB23_split5/AB23_split5.b3d", 1)
        ]

        for file, trial in file_trial_pairs:
            print("Loading: " + file)
            try:
                subject: nimble.biomechanics.SubjectOnDisk = nimble.biomechanics.SubjectOnDisk(file)
                subject.loadAllFrames(doNotStandardizeForcePlateData=True)

                skel: nimble.dynamics.Skeleton = subject.readSkel(0, ignoreGeometry=True)

                trial_protos = subject.getHeaderProto().getTrials()

                trial_len = subject.getTrialLength(trial)
                missing_grf = subject.getMissingGRF(trial)
                track_indices = [missing_reason == nimble.biomechanics.MissingGRFReason.notMissingGRF for missing_reason in missing_grf]

                num_tracked = sum(track_indices)
                if num_tracked == 0 or trial_len < 10:
                    continue

                print("Trial: " + str(trial))
                linear_residuals = np.array(subject.getTrialLinearResidualNorms(trial, 1))

                trial_proto = trial_protos[trial]
                smoothed_pass = trial_proto.getPasses()[1]
                poses = smoothed_pass.getPoses()
                root_poses = poses[3:6, :]
                accs = smoothed_pass.getAccs()
                root_linear_accs = accs[3:6, :]
                # print("Input root linear accs: " + str(np.mean(root_linear_accs, axis=1)))
                com_accs = smoothed_pass.getComAccs()
                # print("Input COM accs: " + str(np.mean(com_accs, axis=1)))
                acc_offsets = com_accs - root_linear_accs

                total_forces = np.zeros((3, trial_len))
                cop_torque_force_in_root = smoothed_pass.getGroundBodyCopTorqueForce()
                num_force_plates = int(cop_torque_force_in_root.shape[0] / 9)
                for j in range(num_force_plates):
                    total_forces += cop_torque_force_in_root[j * 9 + 6:j * 9 + 9, :]

                force_plates = trial_proto.getForcePlates()
                raw_force_data = np.zeros((3, trial_len))
                for plate in force_plates:
                    raw_forces = plate.forces
                    for t in range(trial_len):
                        raw_force_data[:, t] += raw_forces[t]

                # We want our root linear acceleration to offset enough to match the total forces
                goal_com_accs = total_forces / skel.getMass()
                # print("Goal COM accs: " + str(np.mean(goal_com_accs, axis=1)))
                com_acc_errors = goal_com_accs - com_accs
                # print("COM acc errors: " + str(np.mean(com_acc_errors, axis=1)))

                target_root_linear_accs = goal_com_accs - acc_offsets
                # print("Average target root linear acc: " + str(np.mean(target_root_linear_accs, axis=1)))

                # Now we want to try to find a set of root translations that match the target root linear accs on
                # the frames with observed forces, and otherwise revert to our classic
                # AccelerationMinimizingSmoother.

                dt = subject.getTrialTimestep(trial)
                zero_unobserved_acc_weight = 1.0
                track_observed_acc_weight = 10.0
                regularization_weight = 1000.0
                smooth_and_track = nimble.utils.AccelerationTrackAndMinimize(len(track_indices), track_indices, zeroUnobservedAccWeight=zero_unobserved_acc_weight, trackObservedAccWeight=track_observed_acc_weight, regularizationWeight=regularization_weight, dt=dt)

                output_root_poses = np.zeros((3, trial_len))
                for index in range(3):
                    root_pose = root_poses[index, :]
                    target_accs = target_root_linear_accs[index, :]
                    for t in range(trial_len):
                        if not track_indices[t]:
                            target_accs[t] = 0.0
                    output = smooth_and_track.minimize(root_pose, target_accs)
                    output_root_poses[index, :] = output.series
                    offset = output.accelerationOffset
                    print("Axis "+str(index)+" Offset: " + str(offset))

                    input_acc = np.zeros(trial_len)
                    output_acc = np.zeros(trial_len)
                    for t in range(1, trial_len - 1):
                        input_acc[t] = (root_pose[t + 1] - 2 * root_pose[t] + root_pose[t - 1]) / (dt * dt)
                        output_acc[t] = (output.series[t + 1] - 2 * output.series[t] + output.series[t - 1]) / (dt * dt)
                    if trial_len > 2:
                        input_acc[0] = input_acc[1]
                        input_acc[trial_len - 1] = input_acc[trial_len - 2]
                        output_acc[0] = output_acc[1]
                        output_acc[trial_len - 1] = output_acc[trial_len - 2]

                    import matplotlib.pyplot as plt
                    plt.title("Root position axis="+str(index))
                    plt.plot(root_pose, label='Input')
                    plt.plot(output.series, label='Output')
                    plt.legend()
                    plt.show()

                    plt.title("Root acceleration axis="+str(index))
                    plt.plot(input_acc, label='Input')
                    plt.plot(target_accs, label='Target')
                    plt.plot(output_acc, label='Output')
                    plt.plot(total_forces[index, :] / skel.getMass(), label='Total Forces Acc')
                    plt.plot(raw_force_data[index, :] / skel.getMass(), label='Raw Forces Acc')
                    plt.plot(com_accs[index, :], label='Input COM Acc')
                    plt.legend()
                    plt.show()
                    time.sleep(5)

                output_root_acc = np.zeros((3, trial_len))
                for t in range(1, trial_len - 1):
                    output_root_acc[:, t] = (output_root_poses[:, t + 1] - 2 * output_root_poses[:, t] + output_root_poses[:, t - 1]) / (dt * dt)
                if trial_len > 2:
                    output_root_acc[:, 0] = output_root_acc[:, 1]
                    output_root_acc[:, trial_len - 1] = output_root_acc[:, trial_len - 2]
                # print("Output root linear accs: " + str(np.mean(output_root_acc, axis=1)))

                output_com_acc = output_root_acc + acc_offsets
                # print("Output COM accs: " + str(np.mean(output_com_acc, axis=1)))

            except Exception as e:
                print("Error loading: " + file)
                print(e)
                traceback.print_exc()