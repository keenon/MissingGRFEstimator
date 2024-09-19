import random

from cli.abstract_command import AbstractCommand
import os
import nimblephysics as nimble
import numpy as np
import traceback


class Analyze5(AbstractCommand):
    def __init__(self):
        pass

    def register_subcommand(self, subparsers):
        subparser = subparsers.add_parser('analyze5', help='Analyze the distribution of the foot-contact properties on the good GRF data frames.')
        subparser.add_argument('--dataset-home', type=str, default='../data', help='The path to the AddBiomechanics dataset.')

    def run(self, args):
        if 'command' in args and args.command != 'analyze5':
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
        random.shuffle(b3d_files)
        b3d_files = b3d_files[:50]

        treadmill_trials = []
        overground_trials = []

        contact_foot_velocities_overground = []
        contact_foot_velocities_treadmill = []
        non_contact_foot_velocities_overground = []
        non_contact_foot_velocities_treadmill = []

        # Load all the B3D files, and collect statistics for each trial

        for file in b3d_files:
            print("Loading: " + file)
            try:
                subject: nimble.biomechanics.SubjectOnDisk = nimble.biomechanics.SubjectOnDisk(file)

                reviewed_trials = []

                for trial in range(subject.getNumTrials()):
                    missing_reasons = subject.getMissingGRF(trial)
                    manual_review_count = [reason == nimble.biomechanics.MissingGRFReason.manualReview for reason in missing_reasons].count(True)
                    not_missing_count = [reason == nimble.biomechanics.MissingGRFReason.notMissingGRF for reason in missing_reasons].count(True)
                    if manual_review_count + not_missing_count == len(missing_reasons) and not_missing_count > 0:
                        reviewed_trials.append(trial)

                num_dofs = subject.getNumDofs()

                skel = subject.readSkel(0, ignoreGeometry=True)
                ground_body_names = subject.getGroundForceBodies()
                ground_bodies = [skel.getBodyNode(ground_body_name) for ground_body_name in ground_body_names]

                for i, trial in enumerate(reviewed_trials):
                    trial_len = subject.getTrialLength(trial)

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

                    if not any_force:
                        continue

                    # Count the number of steps and the distance traveled by the COP
                    num_steps = 0
                    num_force_plates = subject.getNumForcePlates(trial)
                    last_in_contact = [False for _ in range(num_force_plates)]
                    for t in range(trial_len):
                        frame = frames[t]
                        forces = frame.rawForcePlateForces
                        for f in range(len(forces)):
                            force = forces[f]
                            if np.linalg.norm(force) > 10.0:
                                if not last_in_contact[f]:
                                    last_in_contact[f] = True
                            else:
                                if last_in_contact[f]:
                                    last_in_contact[f] = False
                                    num_steps += 1
                                last_in_contact[f] = False

                    treadmill_trial = (num_steps >= 6) and (num_force_plates == 2)
                    if treadmill_trial:
                        treadmill_trials.append(file+' --trial '+str(trial))
                    else:
                        overground_trials.append(file+' --trial '+str(trial))

                    poses = np.zeros((num_dofs, trial_len))
                    for t in range(trial_len):
                        frame = frames[t]
                        poses[:, t] = frame.processingPasses[0].pos
                    dt = subject.getTrialTimestep(trial)

                    acc_weight = 1.0 / (dt * dt)
                    regularization_weight = 1000.0
                    acc_minimizer = nimble.utils.AccelerationMinimizer(trial_len, acc_weight, regularization_weight)

                    lowpass_poses = np.zeros((num_dofs, trial_len))
                    for i in range(poses.shape[0]):
                        lowpass_poses[i, :] = acc_minimizer.minimize(poses[i, :])
                    poses = lowpass_poses

                    vels = np.zeros((num_dofs, trial_len))
                    for t in range(1, trial_len):
                        vels[:, t] = (poses[:, t] - poses[:, t - 1]) / dt
                    vels[:, 0] = vels[:, 1]

                    for t in range(trial_len):
                        frame = frames[t]

                        if frame.missingGRFReason != nimble.biomechanics.MissingGRFReason.notMissingGRF:
                            continue

                        skel.setPositions(poses[:, t])
                        skel.setVelocities(vels[:, t])

                        ground_body_locations = [body.getWorldTransform().translation() for body in ground_bodies]
                        ground_body_velocities = [body.getLinearVelocity() for body in ground_bodies]

                        bodies_in_contact = [False for _ in ground_bodies]

                        for i in range(len(ground_bodies)):
                            bodies_in_contact[i] = np.linalg.norm(frame.processingPasses[0].groundContactForce[i*3:i*3+3]) > 10.0

                        # forces = frame.rawForcePlateForces
                        # cops = frame.rawForcePlateCenterOfPressures
                        # for f in range(len(forces)):
                        #     force = forces[f]
                        #     if np.linalg.norm(force) > 10.0:
                        #         cop = cops[f]
                        #         closest_ground_body = -1
                        #         closest_ground_body_distance = 1e9
                        #         for i, ground_body_location in enumerate(ground_body_locations):
                        #             if np.linalg.norm(cop - ground_body_location) < closest_ground_body_distance:
                        #                 closest_ground_body = i
                        #                 closest_ground_body_distance = np.linalg.norm(cop - ground_body_location)
                        #         bodies_in_contact[closest_ground_body] = True

                        for i, body in enumerate(ground_bodies):
                            vel = np.linalg.norm(ground_body_velocities[i])
                            if treadmill_trial:
                                if vel > 5.0:
                                    vel = 5.0
                                if bodies_in_contact[i]:
                                    contact_foot_velocities_treadmill.append(vel)
                                else:
                                    non_contact_foot_velocities_treadmill.append(vel)
                            else:
                                if vel > 0.5:
                                    vel = 0.5
                                if bodies_in_contact[i]:
                                    contact_foot_velocities_overground.append(vel)
                                else:
                                    non_contact_foot_velocities_overground.append(vel)

            except Exception as e:
                print("Error loading: " + file)
                print(e)
                traceback.print_exc()

        with open('treadmill_trials.txt', 'w') as f:
            for trial in treadmill_trials:
                f.write('addb view '+trial + '\n')
        with open('overground_trials.txt', 'w') as f:
            for trial in overground_trials:
                f.write('addb view '+trial + '\n')

        # Plot the distribution of the foot velocities
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.hist(contact_foot_velocities_overground, bins=40, alpha=0.5, label='Contact (Overground)')
        ax.hist(non_contact_foot_velocities_overground, bins=40, alpha=0.5, label='Non-contact (Overground)')
        ax.legend()
        plt.show()

        fig, ax = plt.subplots()
        ax.hist(contact_foot_velocities_treadmill, bins=40, alpha=0.5, label='Contact (Treadmill)')
        ax.hist(non_contact_foot_velocities_treadmill, bins=40, alpha=0.5, label='Non-contact (Treadmill)')
        ax.legend()
        plt.show()
