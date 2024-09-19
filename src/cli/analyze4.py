import random

from cli.abstract_command import AbstractCommand
import os
import nimblephysics as nimble
import numpy as np
import traceback


class Analyze4(AbstractCommand):
    def __init__(self):
        pass

    def register_subcommand(self, subparsers):
        subparser = subparsers.add_parser('analyze4', help='Analyze the distribution of the IK performance metrics.')
        subparser.add_argument('--dataset-home', type=str, default='../data', help='The path to the AddBiomechanics dataset.')

    def run(self, args):
        if 'command' in args and args.command != 'analyze4':
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

        bad_ik_max_markers = []
        bad_ik_mean_markers = []
        good_ik_max_markers = []
        good_ik_mean_markers = []

        good_ik_max_markers_dict = {}
        good_ik_mean_markers_dict = {}


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

                for i, trial in enumerate(reviewed_trials):
                    trial_len = subject.getTrialLength(trial)
                    trial_good = trial_good_ik[i]

                    if trial_good:
                        good_ik_max_markers.append(min(np.mean(subject.getTrialMarkerMaxs(trial, 0)), 0.3))
                        good_ik_mean_markers.append(min(np.mean(subject.getTrialMarkerRMSs(trial, 0)), 0.3))

                        good_ik_max_markers_dict[file + ' --trial ' + str(trial)] = min(np.mean(subject.getTrialMarkerMaxs(trial, 0)), 0.3)
                        good_ik_mean_markers_dict[file + ' --trial ' + str(trial)] = min(np.mean(subject.getTrialMarkerRMSs(trial, 0)), 0.3)
                    else:
                        bad_ik_max_markers.append(min(np.mean(subject.getTrialMarkerMaxs(trial, 0)), 0.3))
                        bad_ik_mean_markers.append(min(np.mean(subject.getTrialMarkerRMSs(trial, 0)), 0.3))

            except Exception as e:
                print("Error loading: " + file)
                print(e)
                traceback.print_exc()

        with open('good_ik_max_marker_dist.txt', 'w') as f:
            for key in sorted(good_ik_max_markers_dict.keys(), reverse=True, key=lambda k: good_ik_max_markers_dict[k]):
                f.write(key + ' ' + str(good_ik_max_markers_dict[key]) + '\n')

        with open('good_ik_mean_marker_dist.txt', 'w') as f:
            for key in sorted(good_ik_mean_markers_dict.keys(), reverse=True, key=lambda k: good_ik_mean_markers_dict[k]):
                f.write(key + ' ' + str(good_ik_mean_markers_dict[key]) + '\n')

        print(f"Num glitchy IK: {num_glitchy_ik}")
        print(f"Num trials with no force (excluded from glitchy IK results in what follows): {num_trials_no_force}")
        print(f"Num good IK: {num_good_ik}")
        if len(good_ik_max_markers) > 0:
            print(f"Mean max markers for good IK: {np.mean(good_ik_max_markers)}")
            print(f"Mean mean markers for good IK: {np.mean(good_ik_mean_markers)}")
        if len(bad_ik_max_markers) > 0:
            print(f"Mean max markers for glitchy IK: {np.mean(bad_ik_max_markers)}")
            print(f"Mean mean markers for glitchy IK: {np.mean(bad_ik_mean_markers)}")

        # Plot the histograms
        import matplotlib.pyplot as plt
        plt.hist(good_ik_max_markers, bins=100, alpha=0.5, label='Good IK Max Markers')
        plt.hist(bad_ik_max_markers, bins=100, alpha=0.5, label='Glitchy IK Max Markers')
        plt.legend(loc='upper right')
        plt.show()

        plt.hist(good_ik_mean_markers, bins=100, alpha=0.5, label='Good IK Mean Markers')
        plt.hist(bad_ik_mean_markers, bins=100, alpha=0.5, label='Glitchy IK Mean Markers')
        plt.legend(loc='upper right')
        plt.show()


