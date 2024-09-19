import random

from cli.abstract_command import AbstractCommand
import os
import nimblephysics as nimble
import numpy as np
import traceback


class Analyze2(AbstractCommand):
    def __init__(self):
        pass

    def register_subcommand(self, subparsers):
        subparser = subparsers.add_parser('analyze2', help='Analyze the distribution of the glitchy IK data.')
        subparser.add_argument('--dataset-home', type=str, default='../data', help='The path to the AddBiomechanics dataset.')

    def run(self, args):
        if 'command' in args and args.command != 'analyze2':
            return False
        dataset_home: str = args.dataset_home
        print(f"Running evaluation on {dataset_home}")

        data_dir = os.path.abspath('../data/processed')

        # Recursively list all the B3D files under data
        b3d_files = []
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith(".b3d"):
                    b3d_files.append(os.path.join(root, file))
        # random.shuffle(b3d_files)
        # b3d_files = b3d_files[:10]

        # Load all the B3D files, and collect statistics for each trial

        num_glitchy_ik = 0
        num_good_ik = 0
        num_trials_no_force = 0

        bad_ik_max_vels = []
        good_ik_max_vels = []

        bad_ik_max_lowpass_vels = []
        good_ik_max_lowpass_vels = []

        good_ik_max_vel_trials = {}
        good_ik_max_lowpass_vel_trials = {}
        bad_ik_max_vel_trials = {}
        bad_ik_max_lowpass_vel_trials = {}

        for file in b3d_files:
            print("Loading: " + file)
            try:
                subject: nimble.biomechanics.SubjectOnDisk = nimble.biomechanics.SubjectOnDisk(file)

                reviewed_trials = []
                trial_good_ik = []

                for trial in range(subject.getNumTrials()):
                    missing_reasons = subject.getMissingGRF(trial)
                    manual_review_count = [reason == nimble.biomechanics.MissingGRFReason.manualReview for reason in missing_reasons].count(True)
                    not_missing_count = [reason == nimble.biomechanics.MissingGRFReason.notMissingGRF for reason in missing_reasons].count(True)
                    if manual_review_count + not_missing_count == len(missing_reasons):
                        reviewed_trials.append(trial)
                        if not_missing_count == 0:
                            num_glitchy_ik += 1
                            trial_good_ik.append(False)
                        else:
                            num_good_ik += 1
                            trial_good_ik.append(True)

                num_dofs = subject.getNumDofs()

                for i, trial in enumerate(reviewed_trials):
                    trial_len = subject.getTrialLength(trial)
                    trial_good = trial_good_ik[i]

                    frames = subject.readFrames(trial, 0, trial_len, includeSensorData=True, includeProcessingPasses=True)

                    any_force = False
                    for t in range(trial_len):
                        frame = frames[t]
                        forces = frame.rawForcePlateForces
                        total_force_mag = 0.0
                        for force in forces:
                            total_force_mag += np.linalg.norm(force)
                        if total_force_mag > 10.0:
                            any_force = True
                            break

                    poses = np.zeros((num_dofs, trial_len))
                    for t in range(trial_len):
                        frame = frames[t]
                        poses[:, t] = frame.processingPasses[0].pos
                    dt = subject.getTrialTimestep(trial)

                    vels = np.zeros((num_dofs, trial_len))
                    for t in range(1, trial_len):
                        vels[:, t] = (poses[:, t] - poses[:, t - 1]) / dt
                    vels[:, 0] = vels[:, 1]

                    acc_weight = 1.0 / (dt * dt)
                    regularization_weight = 1000.0
                    acc_minimizer = nimble.utils.AccelerationMinimizer(trial_len, acc_weight, regularization_weight)

                    lowpass_poses = np.zeros((num_dofs, trial_len))
                    for i in range(poses.shape[0]):
                        lowpass_poses[i, :] = acc_minimizer.minimize(poses[i, :])

                    lowpass_vels = np.zeros((num_dofs, trial_len))
                    for t in range(1, trial_len):
                        lowpass_vels[:, t] = (lowpass_poses[:, t] - lowpass_poses[:, t - 1]) / dt
                    lowpass_vels[:, 0] = lowpass_vels[:, 1]

                    max_vel = np.max(np.abs(vels))
                    max_lowpass_vel = np.max(np.abs(lowpass_vels))

                    if trial_good:
                        good_ik_max_vels.append(max_vel)
                        good_ik_max_lowpass_vels.append(max_lowpass_vel)

                        good_ik_max_vel_trials[file + ' ' + str(trial)] = max_vel
                        good_ik_max_lowpass_vel_trials[file + ' ' + str(trial)] = max_lowpass_vel
                    else:
                        # Just exclude the trials without any force at all, since those are trivial
                        if any_force:
                            bad_ik_max_vels.append(max_vel)
                            bad_ik_max_lowpass_vels.append(max_lowpass_vel)

                            bad_ik_max_vel_trials[file + ' ' + str(trial)] = max_vel
                            bad_ik_max_lowpass_vel_trials[file + ' ' + str(trial)] = max_lowpass_vel
                        else:
                            num_trials_no_force += 1

            except Exception as e:
                print("Error loading: " + file)
                print(e)
                traceback.print_exc()

        print(f"Num glitchy IK: {num_glitchy_ik}")
        print(f"Num trials with no force (excluded from glitchy IK results in what follows): {num_trials_no_force}")
        print(f"Num good IK: {num_good_ik}")
        if len(good_ik_max_vels) > 0:
            print(f"Mean max vel for good IK: {np.mean(good_ik_max_vels)}")
            print(f"Min max vel for good IK: {np.min(good_ik_max_vels)}")
        if len(bad_ik_max_vels) > 0:
            print(f"Mean max vel for bad IK: {np.mean(bad_ik_max_vels)}")
            print(f"Min max vel for bad IK: {np.min(bad_ik_max_vels)}")
        if len(good_ik_max_lowpass_vels) > 0:
            print(f"Mean max lowpass vel for good IK: {np.mean(good_ik_max_lowpass_vels)}")
            print(f"Min max lowpass vel for good IK: {np.min(good_ik_max_lowpass_vels)}")
        if len(bad_ik_max_lowpass_vels) > 0:
            print(f"Mean max lowpass vel for bad IK: {np.mean(bad_ik_max_lowpass_vels)}")
            print(f"Min max lowpass vel for bad IK: {np.min(bad_ik_max_lowpass_vels)}")

        with open('good_ik_max_vel_trials.txt', 'w') as f:
            for key in sorted(good_ik_max_vel_trials.keys(), reverse=True, key=lambda k: good_ik_max_vel_trials[k]):
                f.write(key + ' ' + str(good_ik_max_vel_trials[key]) + '\n')

        with open('good_ik_max_lowpass_vel_trials.txt', 'w') as f:
            for key in sorted(good_ik_max_lowpass_vel_trials.keys(), reverse=True, key=lambda k: good_ik_max_lowpass_vel_trials[k]):
                f.write(key + ' ' + str(good_ik_max_lowpass_vel_trials[key]) + '\n')

        with open('bad_ik_max_vel_trials.txt', 'w') as f:
            for key in sorted(bad_ik_max_vel_trials.keys(), reverse=True, key=lambda k: bad_ik_max_vel_trials[k]):
                f.write(key + ' ' + str(bad_ik_max_vel_trials[key]) + '\n')

        with open('bad_ik_max_lowpass_vel_trials.txt', 'w') as f:
            for key in sorted(bad_ik_max_lowpass_vel_trials.keys(), reverse=True, key=lambda k: bad_ik_max_lowpass_vel_trials[k]):
                f.write(key + ' ' + str(bad_ik_max_lowpass_vel_trials[key]) + '\n')

        # Plot the histograms
        import matplotlib.pyplot as plt
        plt.hist(good_ik_max_vels, bins=100, alpha=0.5, label='Good IK Max Vels')
        plt.hist(bad_ik_max_vels, bins=100, alpha=0.5, label='Glitchy IK Max Vels')
        plt.legend(loc='upper right')
        plt.show()

        plt.hist(good_ik_max_lowpass_vels, bins=100, alpha=0.5, label='Good IK Max Lowpass Vels')
        plt.hist(bad_ik_max_lowpass_vels, bins=100, alpha=0.5, label='Glitchy IK Max Lowpass Vels')
        plt.legend(loc='upper right')
        plt.show()
