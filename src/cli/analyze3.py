import random

from cli.abstract_command import AbstractCommand
import os
import nimblephysics as nimble
import numpy as np
import traceback


class Analyze3(AbstractCommand):
    def __init__(self):
        pass

    def register_subcommand(self, subparsers):
        subparser = subparsers.add_parser('analyze3', help='Analyze the distribution of the CoP distances from ground contact bodies in IK data.')
        subparser.add_argument('--dataset-home', type=str, default='../data', help='The path to the AddBiomechanics dataset.')

    def run(self, args):
        if 'command' in args and args.command != 'analyze3':
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
        # b3d_files = b3d_files[:30]

        # Load all the B3D files, and collect statistics for each trial

        num_glitchy_ik = 0
        num_good_ik = 0
        num_trials_no_force = 0

        bad_ik_max_cop_distances = []
        bad_ik_median_cop_distances = []

        good_ik_max_cop_distances = []
        good_ik_median_cop_distances = []

        cop_travel_distances = {}


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

                skel = subject.readSkel(0, ignoreGeometry=True)
                ground_body_names = subject.getGroundForceBodies()
                ground_bodies = [skel.getBodyNode(ground_body_name) for ground_body_name in ground_body_names]

                for i, trial in enumerate(reviewed_trials):
                    trial_len = subject.getTrialLength(trial)
                    trial_good = trial_good_ik[i]

                    frames = subject.readFrames(trial, 0, trial_len, includeSensorData=True, includeProcessingPasses=True)

                    cop_distances = []
                    cop_travel_vectors = []
                    num_steps = 0

                    # cop_elementwise_mins = np.zeros(3)
                    # cop_elementwise_maxs = np.zeros(3)
                    # root_pos_elementwise_mins = np.zeros(3)
                    # root_pos_elementwise_maxs = np.zeros(3)

                    num_force_plates = subject.getNumForcePlates(trial)
                    last_in_contact = [False for _ in range(num_force_plates)]
                    start_cop = [np.zeros(3) for _ in range(num_force_plates)]
                    last_cop = [np.zeros(3) for _ in range(num_force_plates)]
                    for t in range(trial_len):
                        frame = frames[t]
                        skel.setPositions(frame.processingPasses[0].pos)

                        ground_body_locations = [body.getWorldTransform().translation() for body in ground_bodies]

                        forces = frame.rawForcePlateForces
                        cops = frame.rawForcePlateCenterOfPressures
                        for f in range(len(forces)):
                            force = forces[f]
                            if np.linalg.norm(force) > 10.0:
                                cop = cops[f]
                                if not last_in_contact[f]:
                                    start_cop[f] = cop
                                    last_in_contact[f] = True
                                last_cop[f] = cop
                                min_cop_distance = min([np.linalg.norm(cop - ground_body_location) for ground_body_location in ground_body_locations])
                                # Cap at 1 meter, since beyond that is certainly a glitch
                                if min_cop_distance < 1.0:
                                    cop_distances.append(min_cop_distance)
                                else:
                                    cop_distances.append(1.0)
                            else:
                                if last_in_contact[f]:
                                    end_cop = last_cop[f]
                                    last_in_contact[f] = False
                                    if np.linalg.norm(end_cop - start_cop[f]) < 2.0:
                                        cop_travel_vectors.append(end_cop - start_cop[f])
                                        num_steps += 1
                                    start_cop[f] = np.zeros(3)
                                last_in_contact[f] = False

                    if len(cop_distances) > 0:
                        if trial_good:
                            good_ik_max_cop_distances.append(max(cop_distances))
                            good_ik_median_cop_distances.append(np.median(cop_distances))
                        else:
                            bad_ik_max_cop_distances.append(max(cop_distances))
                            bad_ik_median_cop_distances.append(np.median(cop_distances))
                    if trial_good and len(cop_travel_vectors) > 0 and num_force_plates == 2:
                        average_travel_vector = np.mean(cop_travel_vectors, axis=0)
                        cop_travel_distances[file + ' --trial ' + str(trial)] = np.linalg.norm(average_travel_vector)


            except Exception as e:
                print("Error loading: " + file)
                print(e)
                traceback.print_exc()

        with open('cop_travel_distances.txt', 'w') as f:
            for key in sorted(cop_travel_distances.keys(), reverse=True, key=lambda k: cop_travel_distances[k]):
                f.write(key + ' ' + str(cop_travel_distances[key]) + '\n')

        print(f"Num glitchy IK: {num_glitchy_ik}")
        print(f"Num trials with no force (excluded from glitchy IK results in what follows): {num_trials_no_force}")
        print(f"Num good IK: {num_good_ik}")
        if len(good_ik_max_cop_distances) > 0:
            print(f"Mean max CoP distance for good IK: {np.mean(good_ik_max_cop_distances)}")
            print(f"Mean median CoP distance for good IK: {np.mean(good_ik_median_cop_distances)}")
        if len(bad_ik_max_cop_distances) > 0:
            print(f"Mean max CoP distance for glitchy IK: {np.mean(bad_ik_max_cop_distances)}")
            print(f"Mean median CoP distance for glitchy IK: {np.mean(bad_ik_median_cop_distances)}")

        # Plot the histograms
        import matplotlib.pyplot as plt
        # Plot CoP travel distances histogram
        plt.hist(cop_travel_distances.values(), bins=100)
        plt.show()

        plt.hist(good_ik_median_cop_distances, bins=100, alpha=0.5, label='Good IK Median CoP distances')
        plt.hist(bad_ik_median_cop_distances, bins=100, alpha=0.5, label='Glitchy IK Median CoP distances')
        plt.legend(loc='upper right')
        plt.show()

        plt.hist(good_ik_max_cop_distances, bins=100, alpha=0.5, label='Good IK Max CoP distances')
        plt.hist(bad_ik_max_cop_distances, bins=100, alpha=0.5, label='Glitchy IK Max CoP distances')
        plt.legend(loc='upper right')
        plt.show()
