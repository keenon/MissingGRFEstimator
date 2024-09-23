import time
import random
from cli.abstract_command import AbstractCommand
import os
import nimblephysics as nimble
import numpy as np
import traceback


class GRFAnalyze8(AbstractCommand):
    def __init__(self):
        pass

    def register_subcommand(self, subparsers):
        subparser = subparsers.add_parser('grf-analyze8', help='Test a simple least squares solver to minimize angular residuals.')
        subparser.add_argument('--dataset-home', type=str, default='../data', help='The path to the AddBiomechanics dataset.')

    def run(self, args):
        if 'command' in args and args.command != 'grf-analyze8':
            return False
        dataset_home: str = args.dataset_home
        print(f"Running evaluation on {dataset_home}")

        data_dir = os.path.abspath('../data/smoothed')

        # Recursively list all the B3D files under data
        b3d_files = []
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith(".b3d"):
                    file_path = os.path.join(root, file)
                    if 'Santos' in file_path:
                        continue
                    # if 'Carter' in file_path or 'Camargo' in file_path:
                    b3d_files.append(file_path)
        random.seed(42)
        random.shuffle(b3d_files)
        b3d_files = b3d_files[:10]

        # Load all the B3D files, and collect statistics for each trial

        trial_implied_linear_residuals = []
        trial_output_linear_residuals = []
        trial_root_distance = []
        trial_x_acc_offset = []
        trial_y_acc_offset = []
        trial_z_acc_offset = []

        for file in b3d_files:
            print("Loading: " + file)
            try:
                subject: nimble.biomechanics.SubjectOnDisk = nimble.biomechanics.SubjectOnDisk(file)
                subject.loadAllFrames(doNotStandardizeForcePlateData=True)

                skel: nimble.dynamics.Skeleton = subject.readSkel(0, ignoreGeometry=True)

                trial_protos = subject.getHeaderProto().getTrials()

                force_body_indices = [skel.getBodyNode(body_name).getIndexInSkeleton() for body_name in subject.getGroundForceBodies()]
                foot_body_names_recovered = [skel.getBodyNode(i).getName() for i in force_body_indices]
                assert foot_body_names_recovered == subject.getGroundForceBodies()
                residual_helper: nimble.biomechanics.ResidualForceHelper = nimble.biomechanics.ResidualForceHelper(skel, force_body_indices)

                for trial in range(subject.getNumTrials()):
                    trial_len = subject.getTrialLength(trial)
                    missing_grf = subject.getMissingGRF(trial)
                    track_indices = [missing_reason == nimble.biomechanics.MissingGRFReason.notMissingGRF for missing_reason in missing_grf]

                    num_tracked = sum(track_indices)
                    if num_tracked == 0 or trial_len < 10:
                        continue

                    print("Trial: " + str(trial))

                    trial_proto = trial_protos[trial]
                    smoothed_pass = trial_proto.getPasses()[1]
                    poses = smoothed_pass.getPoses()
                    vels = smoothed_pass.getVels()
                    accs = smoothed_pass.getAccs()
                    body_forces = smoothed_pass.getGroundBodyWrenches()

                    root_poses = poses[:3, :]
                    target_root_angular_accs = np.zeros((3, trial_len))
                    for i in range(trial_len):
                        root_acc = residual_helper.calculateResidualFreeRootAcceleration(poses[:, i], vels[:, i], accs[:, i], body_forces[:, i])
                        target_root_angular_accs[:, i] = root_acc[:3]

                    output_root_poses = np.zeros((3, trial_len))

                    dt = subject.getTrialTimestep(trial)
                    zero_unobserved_acc_weight = 0.1
                    track_observed_acc_weight = 100.0
                    regularization_weight = 1000.0
                    smooth_and_track = nimble.utils.AccelerationTrackAndMinimize(len(track_indices), track_indices, zeroUnobservedAccWeight=zero_unobserved_acc_weight, trackObservedAccWeight=track_observed_acc_weight, regularizationWeight=regularization_weight, dt=dt)

                    for index in range(3):
                        root_pose = root_poses[index, :]
                        target_accs = target_root_angular_accs[index, :]
                        for t in range(trial_len):
                            if not track_indices[t]:
                                target_accs[t] = 0.0
                        output = smooth_and_track.minimize(root_pose, target_accs)
                        output_root_poses[index, :] = output.series
                        offset = output.accelerationOffset
                        if index == 0:
                            trial_x_acc_offset.append(offset)
                        elif index == 1:
                            trial_y_acc_offset.append(offset)
                        elif index == 2:
                            trial_z_acc_offset.append(offset)

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

                        # import matplotlib.pyplot as plt
                        # plt.title("Root translation axis="+str(index))
                        # plt.plot(input_acc, label='Input')
                        # plt.plot(target_accs, label='Target')
                        # plt.plot(output_acc, label='Output')
                        # plt.legend()
                        # # plt.plot(root_pose)
                        # # plt.plot(output)
                        # plt.show()
                        # time.sleep(5)

                    output_root_vel = np.zeros((3, trial_len))
                    output_root_acc = np.zeros((3, trial_len))
                    for t in range(1, trial_len - 1):
                        output_root_vel[:, t] = (output_root_poses[:, t] - output_root_poses[:, t - 1]) / dt
                        output_root_acc[:, t] = (output_root_poses[:, t + 1] - 2 * output_root_poses[:, t] + output_root_poses[:, t - 1]) / (dt * dt)
                    if trial_len > 2:
                        output_root_acc[:, 0] = output_root_acc[:, 1]
                        output_root_acc[:, trial_len - 1] = output_root_acc[:, trial_len - 2]
                    # print("Output root linear accs: " + str(np.mean(output_root_acc, axis=1)))

                    # print("Output COM accs: " + str(np.mean(output_com_acc, axis=1)))

                    average_root_offset_distance = np.mean(np.linalg.norm(output_root_poses - root_poses, axis=0))
                    print("Average root offset distance: " + str(average_root_offset_distance))
                    trial_root_distance.append(min(average_root_offset_distance, 0.3))

                    num_timesteps_with_force = 0
                    total_implied_residual = 0.0
                    total_output_residual = 0.0
                    for t in range(len(missing_grf)):
                        if missing_grf[t] == nimble.biomechanics.MissingGRFReason.notMissingGRF:
                            implied_residual = np.linalg.norm(residual_helper.calculateResidual(poses[:, t], vels[:, t], accs[:, t], body_forces[:, t])[:3])

                            updated_pos = np.copy(poses[:, t])
                            updated_pos[:3] = output_root_poses[:, t]
                            updated_vel = np.copy(vels[:, t])
                            updated_vel[:3] = output_root_vel[:, t]
                            updated_acc = np.copy(accs[:, t])
                            updated_acc[:3] = output_root_acc[:, t]

                            output_residual = np.linalg.norm(residual_helper.calculateResidual(updated_pos, updated_vel, updated_acc, body_forces[:, t])[:3])

                            total_implied_residual += implied_residual
                            total_output_residual += output_residual
                            num_timesteps_with_force += 1

                    if num_timesteps_with_force > 0:
                        print("Implied angular residual: " + str(total_implied_residual / num_timesteps_with_force))
                        print("Output angular residual: " + str(total_output_residual / num_timesteps_with_force))
                        trial_implied_linear_residuals.append(min(total_implied_residual / num_timesteps_with_force, 700.0))
                        trial_output_linear_residuals.append(min(total_output_residual / num_timesteps_with_force, 700.0))
            except Exception as e:
                print("Error loading: " + file)
                print(e)
                traceback.print_exc()

        print("Implied linear residuals mean: " + str(np.mean(trial_implied_linear_residuals)))
        print("Output linear residuals mean: " + str(np.mean(trial_output_linear_residuals)))
        print("Root distance mean: " + str(np.mean(trial_root_distance)))
        print("Absolute X acc offset mean: " + str(np.mean(np.abs(trial_x_acc_offset))))
        print("Absolute Y acc offset mean: " + str(np.mean(np.abs(trial_y_acc_offset))))
        print("Absolute Z acc offset mean: " + str(np.mean(np.abs(trial_z_acc_offset))))

        import matplotlib.pyplot as plt
        plt.hist(trial_implied_linear_residuals, bins=20, alpha=0.5, label='Implied')
        plt.hist(trial_output_linear_residuals, bins=20, alpha=0.5, label='Output')
        plt.legend()
        plt.title("Compared Linear Residuals")
        plt.show()

        plt.hist(trial_root_distance, bins=20)
        plt.title("Root Distance")
        plt.show()

        plt.hist(trial_x_acc_offset, bins=20)
        plt.title("X acc offset")
        plt.show()

        plt.hist(trial_y_acc_offset, bins=20)
        plt.title("Y acc offset")
        plt.show()

        plt.hist(trial_z_acc_offset, bins=20)
        plt.title("Z acc offset")
        plt.show()
