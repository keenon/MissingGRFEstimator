import random

from cli.abstract_command import AbstractCommand
import os
import nimblephysics as nimble
import numpy as np
import traceback
import json
from typing import List, Dict, Any, Tuple, Optional
import re
import pandas as pd


class Analyze8(AbstractCommand):
    def __init__(self):
        pass

    def register_subcommand(self, subparsers):
        subparser = subparsers.add_parser('analyze8', help='Look for patterns in bad GRF data.')
        subparser.add_argument('--dataset-home', type=str, default='../data', help='The path to the AddBiomechanics dataset.')
        subparser.add_argument('--foot-marker-file', type=str, default='../foot_marker_file.json', help='The JSON file containing the locations of the markers on the foot.')

    def run(self, args):
        if 'command' in args and args.command != 'analyze8':
            return False
        dataset_home: str = args.dataset_home
        print(f"Running evaluation on {dataset_home}")

        data_dir = os.path.abspath('../data/processed')
        foot_marker_file: str = args.foot_marker_file

        # Recursively list all the B3D files under data
        b3d_files = []
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith(".b3d"):
                    b3d_files.append(os.path.join(root, file))
        # random.shuffle(b3d_files)
        # b3d_files = b3d_files[:5]

        with open(foot_marker_file) as f:
            foot_marker_data = json.load(f)


        good_min_weighted_distance = []
        good_max_weighted_distance = []
        bad_min_weighted_distance = []
        bad_max_weighted_distance = []

        good_mass_weighted_distance = []
        bad_mass_weighted_distance = []

        trials_by_largest_min_weighted_distance = {}
        trials_by_mass_weighted_distance = {}

        for file in b3d_files:
            # if "AB23_split5/AB23_split5.b3d" not in file:
            #     continue

            print("Loading: " + file)
            try:
                subject: nimble.biomechanics.SubjectOnDisk = nimble.biomechanics.SubjectOnDisk(file)
                osim = subject.readOpenSimFile(0, ignoreGeometry=True)

                left_foot_markers = []
                right_foot_markers = []
                for marker_name in foot_marker_data:
                    marker = foot_marker_data[marker_name]
                    mesh_patterns: List[str] = marker['mesh_patterns']
                    mesh_name: Optional[str] = None
                    for mesh in osim.meshMap:
                        if any([pattern in mesh for pattern in mesh_patterns]):
                            mesh_name = mesh
                            break
                    if mesh_name is None:
                        print(f"Could not find mesh for marker {marker_name}")
                        continue
                    offset: np.ndarray = np.array(marker['offset'])
                    body_name: str = osim.meshMap[mesh_name][0]
                    relative_t: nimble.math.Isometry3 = osim.meshMap[mesh_name][1]
                    body: nimble.dynamics.BodyNode = osim.skeleton.getBodyNode(body_name)
                    body_offset: np.ndarray = relative_t.multiply(offset)
                    if marker_name[0] == 'L':
                        left_foot_markers.append((body, body_offset))
                    elif marker_name[0] == 'R':
                        right_foot_markers.append((body, body_offset))

                skel = osim.skeleton
                foot_markers = [left_foot_markers, right_foot_markers]

                trial_bad = 'Tan2021' in file

                for trial in range(subject.getNumTrials()):
                    # if trial != 46:
                    #     continue

                    # Load the trial and smooth it
                    trial_len = subject.getTrialLength(trial)
                    frames = subject.readFrames(trial, 0, trial_len, includeSensorData=True, includeProcessingPasses=True)
                    num_dofs = subject.getNumDofs()
                    poses = np.zeros((num_dofs, trial_len))
                    for t in range(trial_len):
                        frame = frames[t]
                        poses[:, t] = frame.processingPasses[0].pos
                    dt = subject.getTrialTimestep(trial)

                    # acc_weight = 1.0 / (dt * dt)
                    # regularization_weight = 1000.0
                    # acc_minimizer = nimble.utils.AccelerationMinimizer(trial_len, acc_weight, regularization_weight, numIterations=1000)
                    #
                    # lowpass_poses = np.zeros((num_dofs, trial_len))
                    # for i in range(poses.shape[0]):
                    #     lowpass_poses[i, :] = acc_minimizer.minimize(poses[i, :])
                    #
                    # vels = np.zeros((num_dofs, trial_len))
                    # for t in range(1, trial_len):
                    #     vels[:, t] = (lowpass_poses[:, t] - lowpass_poses[:, t - 1]) / dt
                    # vels[:, 0] = vels[:, 1]
                    # poses = lowpass_poses

                    # Compute the foot travel distance while in contact

                    num_contact_bodies = len(foot_markers)
                    num_force_plates = subject.getNumForcePlates(trial)
                    last_in_contact = [False for _ in range(num_force_plates)]

                    contact_distances = []
                    contact_forces = []
                    for _ in range(num_force_plates):
                        contact_distances.append([0.0 for _ in range(num_contact_bodies)])
                        contact_forces.append(0.0)

                    largest_min_weighted_distance = 0.0
                    mass_weighted_distance = 0.0
                    total_force = 0.0

                    for t in range(trial_len):
                        frame = frames[t]
                        skel.setPositions(frame.processingPasses[0].pos)

                        forces = frame.rawForcePlateForces
                        cops = frame.rawForcePlateCenterOfPressures
                        for f in range(len(forces)):
                            force = forces[f]
                            force_mag = np.linalg.norm(force)
                            if force_mag > 10.0:
                                cop = cops[f]
                                if not last_in_contact[f]:
                                    last_in_contact[f] = True
                                for b in range(num_contact_bodies):
                                    marker_positions = skel.getMarkerWorldPositions(foot_markers[b])
                                    marker_positions_as_3vecs = []
                                    for i in range(int(len(marker_positions) / 3)):
                                        marker_positions_as_3vecs.append(marker_positions[i * 3:i * 3 + 3])
                                    dist = nimble.math.distancePointToConvexHullProjectedTo2D(cop, marker_positions_as_3vecs, [0.0, 1.0, 0.0])
                                    contact_distances[f][b] += dist * force_mag
                                    contact_forces[f] += force_mag
                            else:
                                if last_in_contact[f]:
                                    if contact_forces[f] > 100.0:
                                        weighted_average_distances = [contact_distances[f][body] / contact_forces[f] for body in range(num_contact_bodies)]
                                        min_weighted_distance = min(weighted_average_distances)
                                        max_weighted_distance = max(weighted_average_distances)
                                        if min_weighted_distance > largest_min_weighted_distance:
                                            largest_min_weighted_distance = min_weighted_distance
                                        mass_weighted_distance += min_weighted_distance * contact_forces[f]
                                        total_force += contact_forces[f]

                                        if trial_bad:
                                            bad_min_weighted_distance.append(min_weighted_distance)
                                            bad_max_weighted_distance.append(max_weighted_distance)
                                        else:
                                            good_min_weighted_distance.append(min_weighted_distance)
                                            good_max_weighted_distance.append(max_weighted_distance)

                                    last_in_contact[f] = False
                                    contact_distances[f] = [0.0 for _ in range(num_contact_bodies)]
                                    contact_forces[f] = 0.0
                                last_in_contact[f] = False
                    for f in range(num_force_plates):
                        if last_in_contact[f]:
                            if contact_forces[f] > 100.0:
                                weighted_average_distances = [contact_distances[f][body] / contact_forces[f] for body in range(num_contact_bodies)]
                                min_weighted_distance = min(weighted_average_distances)
                                max_weighted_distance = max(weighted_average_distances)
                                if min_weighted_distance > largest_min_weighted_distance:
                                    largest_min_weighted_distance = min_weighted_distance
                                mass_weighted_distance += min_weighted_distance * contact_forces[f]
                                total_force += contact_forces[f]

                                if trial_bad:
                                    bad_min_weighted_distance.append(min_weighted_distance)
                                    bad_max_weighted_distance.append(max_weighted_distance)
                                else:
                                    good_min_weighted_distance.append(min_weighted_distance)
                                    good_max_weighted_distance.append(max_weighted_distance)

                            last_in_contact[f] = False
                            contact_distances[f] = [0.0 for _ in range(num_contact_bodies)]
                            contact_forces[f] = 0.0

                    if largest_min_weighted_distance > 0.0:
                        trials_by_largest_min_weighted_distance['addb view '+file + ' --trial ' + str(trial)] = largest_min_weighted_distance
                    if total_force > 0.0:
                        trials_by_mass_weighted_distance['addb view '+file + ' --trial ' + str(trial)] = mass_weighted_distance / total_force
                        if trial_bad:
                            bad_mass_weighted_distance.append(mass_weighted_distance / total_force)
                        else:
                            good_mass_weighted_distance.append(mass_weighted_distance / total_force)

            except Exception as e:
                print("Error loading: " + file)
                print(e)
                traceback.print_exc()

        with open('trials_by_largest_min_weighted_distance.txt', 'w') as f:
            for key in sorted(trials_by_largest_min_weighted_distance.keys(), reverse=True, key=lambda k: trials_by_largest_min_weighted_distance[k]):
                f.write(key + ' ' + str(trials_by_largest_min_weighted_distance[key]) + '\n')

        with open('trials_by_mass_weighted_distance.txt', 'w') as f:
            for key in sorted(trials_by_mass_weighted_distance.keys(), reverse=True, key=lambda k: trials_by_mass_weighted_distance[k]):
                f.write(key + ' ' + str(trials_by_mass_weighted_distance[key]) + '\n')

        # Plot histograms comparing good and bad on each type of measurement
        import matplotlib.pyplot as plt

        # Plot the histogram on the range 0 to 0.03 on the x axis
        plt.hist(good_mass_weighted_distance, bins=50, alpha=0.5, label='Good', range=(0, 0.03))
        plt.hist(bad_mass_weighted_distance, bins=50, alpha=0.5, label='Bad', range=(0, 0.03))
        plt.title('Mass weighted distance')
        plt.legend(loc='upper right')
        plt.show()

        fig, axs = plt.subplots(1, 2)
        axs[0].hist(good_min_weighted_distance, bins=50, alpha=0.5, label='Good', range=(0, 0.03))
        axs[0].hist(bad_min_weighted_distance, bins=50, alpha=0.5, label='Bad', range=(0, 0.03))
        axs[0].set_title('Min weighted distance')

        axs[1].hist(good_max_weighted_distance, bins=50, alpha=0.5, label='Good', range=(0, 0.5))
        axs[1].hist(bad_max_weighted_distance, bins=50, alpha=0.5, label='Bad', range=(0, 1.0))
        axs[1].set_title('Max weighted distance')

        plt.legend(loc='upper right')

        plt.show()
