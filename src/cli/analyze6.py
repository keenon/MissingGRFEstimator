import random

from cli.abstract_command import AbstractCommand
import os
import nimblephysics as nimble
import numpy as np
import traceback
import json
from typing import List, Dict, Any, Tuple
import re
import pandas as pd


class Analyze6(AbstractCommand):
    def __init__(self):
        pass

    def register_subcommand(self, subparsers):
        subparser = subparsers.add_parser('analyze6', help='Analyze the differences between treadmill and non-treadmill data.')
        subparser.add_argument('--dataset-home', type=str, default='../data', help='The path to the AddBiomechanics dataset.')
        subparser.add_argument('--treadmill-regex-patterns', type=str, default='../treadmill_regex_patterns.json', help='The path to the treadmill regex patterns.')
        subparser.add_argument('--output-featurized-csv', type=str, default='trial_classification_features.csv', help='The path to the output featurized CSV file.')

    def run(self, args):
        if 'command' in args and args.command != 'analyze6':
            return False
        dataset_home: str = args.dataset_home
        treadmill_regex_patterns: str = args.treadmill_regex_patterns
        output_featurized_csv: str = args.output_featurized_csv
        print(f"Running evaluation on {dataset_home}")

        data_dir = os.path.abspath('../data/processed')

        # Recursively list all the B3D files under data
        b3d_files = []
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith(".b3d"):
                    b3d_files.append(os.path.join(root, file))
        # random.shuffle(b3d_files)
        # b3d_files = b3d_files[:20]

        # Open the treadmill regex patterns
        with open(treadmill_regex_patterns) as f:
            treadmill_regex_patterns = json.load(f)
            patterns: List[Dict[str, Any]] = treadmill_regex_patterns['patterns']

        treadmill_trials = []
        static_trials = []
        overground_trials = []
        unclassified_trials = []

        treadmill_max_root_rot_vels = []
        treadmill_max_root_lin_vels = []
        static_max_root_rot_vels = []
        static_max_root_lin_vels = []
        overground_max_root_rot_vels = []
        overground_max_root_lin_vels = []
        unclassified_max_root_rot_vels = []
        unclassified_max_root_lin_vels = []

        treadmill_root_box_volumes = []
        static_root_box_volumes = []
        overground_root_box_volumes = []
        unclassified_root_box_volumes = []

        treadmill_num_force_plates = []
        static_num_force_plates = []
        overground_num_force_plates = []
        unclassified_num_force_plates = []

        treadmill_num_steps = []
        treadmill_num_steps_per_force_plate = []
        static_num_steps = []
        static_num_steps_per_force_plate = []
        overground_num_steps = []
        overground_num_steps_per_force_plate = []
        unclassified_num_steps = []
        unclassified_num_steps_per_force_plate = []

        treadmill_travel_distances = []
        static_travel_distances = []
        overground_travel_distances = []
        unclassified_travel_distances = []

        treadmill_travel_distances_over_box_volume = []
        static_travel_distances_over_box_volume = []
        overground_travel_distances_over_box_volume = []
        unclassified_travel_distances_over_box_volume = []

        # For printing out ordered lists to files
        treadmill_trial_travel_distances = {}
        overground_trial_box_volumes = {}

        num_correct = 0
        total_trials = 0
        errors = []

        featurized_data = []

        for file in b3d_files:
            print("Loading: " + file)
            try:
                subject: nimble.biomechanics.SubjectOnDisk = nimble.biomechanics.SubjectOnDisk(file)
                skel = subject.readSkel(0, ignoreGeometry=True)
                ground_body_names = subject.getGroundForceBodies()
                ground_bodies = [skel.getBodyNode(ground_body_name) for ground_body_name in ground_body_names]

                for trial in range(subject.getNumTrials()):

                    # Ignore trials that haven't been manually reviewed, and where we didn't drop all the frames

                    missing_reasons = subject.getMissingGRF(trial)
                    manual_review_count = [reason == nimble.biomechanics.MissingGRFReason.manualReview for reason in missing_reasons].count(True)
                    not_missing_count = [reason == nimble.biomechanics.MissingGRFReason.notMissingGRF for reason in missing_reasons].count(True)
                    if manual_review_count + not_missing_count != len(missing_reasons) or not_missing_count == 0:
                        continue

                    # Pick out what kind of trial this is, from our manually reviewed regex patterns

                    pattern_treadmill = False
                    pattern_static = False
                    pattern_overground = False
                    pattern_unclassified = False

                    found_match = False
                    for pattern in patterns:
                        pattern_static = False
                        pattern_treadmill = False
                        pattern_overground = False
                        pattern_unclassified = False
                        file_pattern_regex = pattern['file_pattern']
                        if 'treadmill' in pattern:
                            pattern_treadmill = pattern['treadmill']
                        if 'static' in pattern:
                            pattern_static = pattern['static']
                        if 'trial_ge' in pattern:
                            if trial < pattern['trial_ge']:
                                continue
                        file_pattern_match = re.match(file_pattern_regex, file)
                        if file_pattern_match:
                            found_match = True
                            if pattern_treadmill:
                                treadmill_trials.append('addb view ' + file + ' --trial ' + str(trial))
                            elif pattern_static:
                                static_trials.append('addb view ' + file + ' --trial ' + str(trial))
                            else:
                                pattern_overground = True
                                overground_trials.append('addb view ' + file + ' --trial ' + str(trial))
                            break
                    if not found_match:
                        pattern_unclassified = True
                        unclassified_trials.append('addb view ' + file + ' --trial ' + str(trial))
                        continue

                    # Load the trial and smooth it

                    trial_len = subject.getTrialLength(trial)
                    frames = subject.readFrames(trial, 0, trial_len, includeSensorData=True, includeProcessingPasses=True)
                    num_dofs = subject.getNumDofs()
                    poses = np.zeros((num_dofs, trial_len))
                    for t in range(trial_len):
                        frame = frames[t]
                        poses[:, t] = frame.processingPasses[0].pos
                    dt = subject.getTrialTimestep(trial)

                    acc_weight = 1.0 / (dt * dt)
                    regularization_weight = 1000.0
                    acc_minimizer = nimble.utils.AccelerationMinimizer(trial_len, acc_weight, regularization_weight, numIterations=1000)

                    lowpass_poses = np.zeros((num_dofs, trial_len))
                    for i in range(poses.shape[0]):
                        lowpass_poses[i, :] = acc_minimizer.minimize(poses[i, :])

                    vels = np.zeros((num_dofs, trial_len))
                    for t in range(1, trial_len):
                        vels[:, t] = (lowpass_poses[:, t] - lowpass_poses[:, t - 1]) / dt
                    vels[:, 0] = vels[:, 1]
                    poses = lowpass_poses

                    # Compute the max root velocities

                    max_root_rot_vel = np.max(np.abs(vels[0:3, :]))
                    # Cap the root rotational velocity at 5.0 rad/s
                    if max_root_rot_vel > 5.0:
                        max_root_rot_vel = 5.0
                    max_root_lin_vel = np.max(np.abs(vels[3:6, :]))

                    if pattern_treadmill:
                        treadmill_max_root_rot_vels.append(max_root_rot_vel)
                        treadmill_max_root_lin_vels.append(max_root_lin_vel)
                    elif pattern_static:
                        static_max_root_rot_vels.append(max_root_rot_vel)
                        static_max_root_lin_vels.append(max_root_lin_vel)
                    elif pattern_overground:
                        overground_max_root_rot_vels.append(max_root_rot_vel)
                        overground_max_root_lin_vels.append(max_root_lin_vel)
                    else:
                        unclassified_max_root_rot_vels.append(max_root_rot_vel)
                        unclassified_max_root_lin_vels.append(max_root_lin_vel)

                    # Compute the root box volumes

                    root_translation = poses[3:6, :]
                    root_box_lower_bound = np.min(root_translation, axis=1)
                    root_box_upper_bound = np.max(root_translation, axis=1)
                    root_box_volume = np.sum(root_box_upper_bound - root_box_lower_bound)
                    if root_box_volume > 10.0:
                        root_box_volume = 10.0

                    if pattern_treadmill:
                        treadmill_root_box_volumes.append(root_box_volume)
                    elif pattern_static:
                        static_root_box_volumes.append(root_box_volume)
                    elif pattern_overground:
                        overground_root_box_volumes.append(root_box_volume)
                        overground_trial_box_volumes['addb view '+file + ' --trial ' + str(trial)] = root_box_volume
                    else:
                        unclassified_root_box_volumes.append(root_box_volume)

                    # Log the number of force plates

                    num_force_plates = subject.getNumForcePlates(trial)
                    if pattern_treadmill:
                        treadmill_num_force_plates.append(num_force_plates)
                    elif pattern_static:
                        static_num_force_plates.append(num_force_plates)
                    elif pattern_overground:
                        overground_num_force_plates.append(num_force_plates)
                    else:
                        unclassified_num_force_plates.append(num_force_plates)

                    # Compute the number of steps

                    num_steps = 0
                    num_steps_per_force_plate = [0 for _ in range(num_force_plates)]
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
                                    num_steps_per_force_plate[f] += 1
                                last_in_contact[f] = False

                    if pattern_treadmill:
                        treadmill_num_steps.append(num_steps)
                        for f in range(num_force_plates):
                            treadmill_num_steps_per_force_plate.append(num_steps_per_force_plate[f])
                    elif pattern_static:
                        static_num_steps.append(num_steps)
                        for f in range(num_force_plates):
                            static_num_steps_per_force_plate.append(num_steps_per_force_plate[f])
                    elif pattern_overground:
                        overground_num_steps.append(num_steps)
                        for f in range(num_force_plates):
                            overground_num_steps_per_force_plate.append(num_steps_per_force_plate[f])
                    else:
                        unclassified_num_steps.append(num_steps)
                        for f in range(num_force_plates):
                            unclassified_num_steps_per_force_plate.append(num_steps_per_force_plate[f])

                    # Compute the foot travel distance while in contact

                    num_contact_bodies = len(subject.getGroundForceBodies())
                    body_last_in_contact = [False for _ in range(num_contact_bodies)]
                    body_started_contact = [np.zeros(3) for _ in range(num_contact_bodies)]
                    body_last_position = [np.zeros(3) for _ in range(num_contact_bodies)]
                    step_travel_distances = []
                    for t in range(trial_len):
                        frame = frames[t]
                        skel.setPositions(poses[:, t])
                        ground_body_locations = [body.getWorldTransform().translation() for body in ground_bodies]
                        forces = frame.processingPasses[0].groundContactForce
                        for f in range(len(ground_body_locations)):
                            force = forces[f*3:f*3+3]
                            if np.linalg.norm(force) > 10.0:
                                body_last_position[f] = ground_body_locations[f]
                                if not body_last_in_contact[f]:
                                    body_started_contact[f] = ground_body_locations[f]
                                    body_last_in_contact[f] = True
                            else:
                                if body_last_in_contact[f]:
                                    body_last_in_contact[f] = False
                                    step_travel_distances.append(np.linalg.norm(body_last_position[f] - body_started_contact[f]))

                    if pattern_treadmill:
                        treadmill_travel_distances.append(np.max(step_travel_distances))
                        treadmill_trial_travel_distances['addb view '+file + ' --trial ' + str(trial)] = np.max(step_travel_distances)
                        treadmill_travel_distances_over_box_volume.append(np.max(step_travel_distances) / max(root_box_volume, 0.001))
                    elif pattern_static:
                        static_travel_distances.append(np.max(step_travel_distances))
                        static_travel_distances_over_box_volume.append(np.max(step_travel_distances) / max(root_box_volume, 0.001))
                    elif pattern_overground:
                        overground_travel_distances.append(np.max(step_travel_distances))
                        overground_travel_distances_over_box_volume.append(np.max(step_travel_distances) / max(root_box_volume, 0.001))
                    else:
                        unclassified_travel_distances.append(np.max(step_travel_distances))
                        unclassified_travel_distances_over_box_volume.append(np.max(step_travel_distances) / max(root_box_volume, 0.001))

                    pattern_type = 'Treadmill' if pattern_treadmill else 'Static' if pattern_static else 'Overground' if pattern_overground else 'Unclassified'
                    guessed_type = 'Overground'
                    if num_steps > 15 and num_force_plates == 2:
                        guessed_type = 'Treadmill'
                    if np.max(step_travel_distances) > 0.4 and num_force_plates == 2:
                        guessed_type = 'Treadmill'
                    if root_box_volume > 0.8:
                        guessed_type = 'Overground'
                    if root_box_volume < 0.06 or max_root_rot_vel < 0.1:
                        guessed_type = 'Static'

                    if pattern_type == guessed_type:
                        num_correct += 1
                    else:
                        errors.append('addb view '+file + ' --trial ' + str(trial) + ' ::: true=' + pattern_type +
                                      ', guess=' + guessed_type+', max_step_trave_distance='+str(np.max(step_travel_distances))+
                                      ', root_box='+str(root_box_volume)+', max_rot_vel='+str(max_root_rot_vel)+
                                      ', max_lin_vel='+str(max_root_lin_vel)+', num_steps='+str(num_steps)+
                                      ', num_force_plates='+str(num_force_plates)+', max_num_steps_per_force_plate='+str(max(num_steps_per_force_plate, default=0))+
                                        ', min_num_steps_per_force_plate='+str(min(num_steps_per_force_plate, default=0)))
                    total_trials += 1

                    featurized_data.append({
                        'file': file,
                        'trial': trial,
                        'pattern_type': pattern_type,
                        'mean_step_travel_distance': np.mean(step_travel_distances),
                        'max_step_travel_distance': np.max(step_travel_distances),
                        'root_box_volume': root_box_volume,
                        'max_root_rot_vel': max_root_rot_vel,
                        'max_root_lin_vel': max_root_lin_vel,
                        'num_steps': num_steps,
                        'num_force_plates': num_force_plates,
                        'max_num_steps_per_force_plate': max(num_steps_per_force_plate, default=0),
                        'min_num_steps_per_force_plate': min(num_steps_per_force_plate, default=0),
                    })




            except Exception as e:
                print("Error loading: " + file)
                print(e)
                traceback.print_exc()

        print('Counts:')
        print('Treadmill:', len(treadmill_trials))
        print('Static:', len(static_trials))
        print('Overground:', len(overground_trials))
        print('Unclassified:', len(unclassified_trials))

        print('Correct:', num_correct)
        print('Total:', total_trials)
        print('Accuracy:', num_correct / total_trials)

        # Write the featurized data to a CSV file
        df = pd.DataFrame(featurized_data)
        df.to_csv(output_featurized_csv, index=False)

        with open('errors.txt', 'w') as f:
            for error in errors:
                f.write(error + '\n')
        with open('treadmill_trial_travel_distances.txt', 'w') as f:
            for trial in sorted(treadmill_trial_travel_distances, key=treadmill_trial_travel_distances.get, reverse=True):
                f.write(trial + ' ' + str(treadmill_trial_travel_distances[trial]) + '\n')
        with open('overground_trial_box_volumes.txt', 'w') as f:
            for trial in sorted(overground_trial_box_volumes, key=overground_trial_box_volumes.get, reverse=True):
                f.write(trial + ' ' + str(overground_trial_box_volumes[trial]) + '\n')
        with open('treadmill_trials.txt', 'w') as f:
            for trial in treadmill_trials:
                f.write(trial + '\n')
        with open('static_trials.txt', 'w') as f:
            for trial in static_trials:
                f.write(trial + '\n')
        with open('overground_trials.txt', 'w') as f:
            for trial in overground_trials:
                f.write(trial + '\n')
        with open('unclassified_trials.txt', 'w') as f:
            for trial in unclassified_trials:
                f.write(trial + '\n')

        # Plot the result histograms
        import matplotlib.pyplot as plt

        plt.hist(treadmill_travel_distances, bins=50, alpha=0.5, label='Treadmill')
        plt.hist(static_travel_distances, bins=50, alpha=0.5, label='Static')
        plt.hist(overground_travel_distances, bins=50, alpha=0.5, label='Overground')
        plt.hist(unclassified_travel_distances, bins=50, alpha=0.5, label='Unclassified')
        plt.legend(loc='upper right')
        plt.title('Foot travel distances')
        plt.show()

        plt.hist(treadmill_travel_distances_over_box_volume, bins=50, alpha=0.5, label='Treadmill')
        plt.hist(static_travel_distances_over_box_volume, bins=50, alpha=0.5, label='Static')
        plt.hist(overground_travel_distances_over_box_volume, bins=50, alpha=0.5, label='Overground')
        plt.hist(unclassified_travel_distances_over_box_volume, bins=50, alpha=0.5, label='Unclassified')
        plt.legend(loc='upper right')
        plt.title('Foot travel distances over root box volume')
        plt.show()

        plt.hist(treadmill_max_root_rot_vels, bins=50, alpha=0.5, label='Treadmill')
        plt.hist(static_max_root_rot_vels, bins=50, alpha=0.5, label='Static')
        plt.hist(overground_max_root_rot_vels, bins=50, alpha=0.5, label='Overground')
        plt.hist(unclassified_max_root_rot_vels, bins=50, alpha=0.5, label='Unclassified')
        plt.legend(loc='upper right')
        plt.title('Max root rot vel')
        plt.show()

        plt.hist(treadmill_max_root_lin_vels, bins=50, alpha=0.5, label='Treadmill')
        plt.hist(static_max_root_lin_vels, bins=50, alpha=0.5, label='Static')
        plt.hist(overground_max_root_lin_vels, bins=50, alpha=0.5, label='Overground')
        plt.hist(unclassified_max_root_lin_vels, bins=50, alpha=0.5, label='Unclassified')
        plt.legend(loc='upper right')
        plt.title('Max root lin vel')
        plt.show()

        plt.hist(treadmill_root_box_volumes, bins=50, alpha=0.5, label='Treadmill')
        plt.hist(static_root_box_volumes, bins=50, alpha=0.5, label='Static')
        plt.hist(overground_root_box_volumes, bins=50, alpha=0.5, label='Overground')
        plt.hist(unclassified_root_box_volumes, bins=50, alpha=0.5, label='Unclassified')
        plt.legend(loc='upper right')
        plt.title('Root box volumes')
        plt.show()

        max_num_force_plates = max(max(treadmill_num_force_plates, default=0), max(static_num_force_plates, default=0), max(overground_num_force_plates, default=0), max(unclassified_num_force_plates, default=0)) + 1
        plt.hist(treadmill_num_force_plates, bins=max_num_force_plates, alpha=0.5, label='Treadmill')
        plt.hist(static_num_force_plates, bins=max_num_force_plates, alpha=0.5, label='Static')
        plt.hist(overground_num_force_plates, bins=max_num_force_plates, alpha=0.5, label='Overground')
        plt.hist(unclassified_num_force_plates, bins=max_num_force_plates, alpha=0.5, label='Unclassified')
        plt.legend(loc='upper right')
        plt.title('Num force plates')
        plt.show()

        max_num_steps = max(max(treadmill_num_steps, default=0), max(static_num_steps,default=0), max(overground_num_steps, default=0), max(unclassified_num_steps, default=0)) + 1
        plt.hist(treadmill_num_steps, bins=max_num_steps, alpha=0.5, label='Treadmill')
        plt.hist(static_num_steps, bins=max_num_steps, alpha=0.5, label='Static')
        plt.hist(overground_num_steps, bins=max_num_steps, alpha=0.5, label='Overground')
        plt.hist(unclassified_num_steps, bins=max_num_steps, alpha=0.5, label='Unclassified')
        plt.legend(loc='upper right')
        plt.title('Num steps')
        plt.show()

        max_num_steps_per_force_plate = max(max(treadmill_num_steps_per_force_plate, default=0), max(static_num_steps_per_force_plate, default=0), max(overground_num_steps_per_force_plate, default=0), max(unclassified_num_steps_per_force_plate, default=0)) + 1
        plt.hist(treadmill_num_steps_per_force_plate, bins=max_num_steps_per_force_plate, alpha=0.5, label='Treadmill')
        plt.hist(static_num_steps_per_force_plate, bins=max_num_steps_per_force_plate, alpha=0.5, label='Static')
        plt.hist(overground_num_steps_per_force_plate, bins=max_num_steps_per_force_plate, alpha=0.5, label='Overground')
        plt.hist(unclassified_num_steps_per_force_plate, bins=max_num_steps_per_force_plate, alpha=0.5, label='Unclassified')
        plt.legend(loc='upper right')
        plt.title('Num steps per force plate')
        plt.show()
