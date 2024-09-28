from cli.abstract_command import AbstractCommand
import os
import nimblephysics as nimble
from detectors.abstract_detector import AbstractDetector
from detectors.thresholds import ThresholdsDetector
from detectors.thresholds_preloaded import ThresholdsDetectorPreloaded
from detectors.zero_grf import ZeroGRFDetector
import numpy as np
import random
import traceback


class EvaluatePerformance(AbstractCommand):
    def __init__(self):
        pass

    def register_subcommand(self, subparsers):
        subparser = subparsers.add_parser('eval', help='Evaluate the performance of a missing GRF heuristic on the dataset.')
        subparser.add_argument('--dataset-home', type=str, default='../data', help='The path to the AddBiomechanics dataset.')
        subparser.add_argument('--method', type=str, default='threshold-preloaded', help='The method to evaluate.')
        subparser.add_argument('--foot-marker-file', type=str, default='../foot_marker_file.json', help='The JSON file containing the locations of the markers on the foot.')
        subparser.add_argument('--log-csv', type=str, default='../eval.csv', help='The file to log output to.')

    def run(self, args):
        if 'command' in args and args.command != 'eval':
            return False
        dataset_home: str = args.dataset_home
        method: str = args.method
        foot_marker_file: str = args.foot_marker_file
        print(f"Running evaluation on {dataset_home} using method {method}")

        detector = AbstractDetector()
        if method == 'threshold':
            detector = ThresholdsDetector(foot_marker_file)
        if method == 'threshold-preloaded':
            detector = ThresholdsDetectorPreloaded(foot_marker_file)

        data_dir = os.path.abspath('../data/processed')

        # Recursively list all the B3D files under data
        b3d_files = []
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith(".b3d"):
                    path = os.path.join(root, file)
                    # if 'Camargo' in path:
                    b3d_files.append(path)
        random.shuffle(b3d_files)
        b3d_files = b3d_files[:30]

        # Load all the B3D files, and collect statistics for each trial

        num_trials_manually_reviewed = 0
        num_trials_not_manually_reviewed = 0

        num_missing_frames = 0
        num_not_missing_frames = 0

        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0

        true_segment_lengths = []
        predicted_segment_lengths = []

        trials_by_recall = {}
        trials_by_precision = {}
        trials_by_f1 = {}
        trials_by_false_negatives = {}
        trials_by_false_positives = {}

        for file in b3d_files:
            print("Loading: " + file)
            try:
                subject: nimble.biomechanics.SubjectOnDisk = nimble.biomechanics.SubjectOnDisk(file)

                reviewed_trials = []
                reviewed_trial_missing = []

                for trial in range(subject.getNumTrials()):
                    missing_reasons = subject.getMissingGRF(trial)
                    manual_review_count = [reason == nimble.biomechanics.MissingGRFReason.manualReview for reason in missing_reasons].count(True)
                    not_missing_count = [reason == nimble.biomechanics.MissingGRFReason.notMissingGRF for reason in missing_reasons].count(True)
                    if manual_review_count + not_missing_count != len(missing_reasons):
                        num_trials_not_manually_reviewed += 1
                    else:
                        num_trials_manually_reviewed += 1
                        num_missing_frames += manual_review_count
                        num_not_missing_frames += not_missing_count
                        reviewed_trials.append(trial)
                        reviewed_trial_missing.append([reason != nimble.biomechanics.MissingGRFReason.notMissingGRF for reason in missing_reasons])

                # Run the detector
                estimated_missing = detector.estimate_missing_grfs(subject, reviewed_trials)

                for i in range(len(reviewed_trials)):
                    trial_true_positives = 0
                    trial_false_positives = 0
                    trial_true_negatives = 0
                    trial_false_negatives = 0

                    for j in range(len(estimated_missing[i])):
                        if estimated_missing[i][j]:
                            if reviewed_trial_missing[i][j]:
                                trial_true_positives += 1
                                true_positives += 1
                            else:
                                trial_false_positives += 1
                                false_positives += 1
                        else:
                            if reviewed_trial_missing[i][j]:
                                trial_false_negatives += 1
                                false_negatives += 1
                            else:
                                trial_true_negatives += 1
                                true_negatives += 1

                    trial_precision = 0
                    trial_recall = 0
                    trial_f1 = 0
                    if trial_true_positives + trial_false_positives != 0:
                        trial_precision = trial_true_positives / (trial_true_positives + trial_false_positives)
                        trials_by_precision['addb view '+file + ' --trial ' + str(reviewed_trials[i])] = trial_precision
                    if trial_true_positives + trial_false_negatives != 0:
                        trial_recall = trial_true_positives / (trial_true_positives + trial_false_negatives)
                        trials_by_recall['addb view ' + file + ' --trial ' + str(reviewed_trials[i])] = trial_recall
                    if trial_precision + trial_recall != 0:
                        trial_f1 = 2 * trial_precision * trial_recall / (trial_precision + trial_recall)
                        trials_by_f1['addb view '+file + ' --trial ' + str(reviewed_trials[i])] = trial_f1

                    trials_by_false_negatives['addb view '+file + ' --trial ' + str(reviewed_trials[i])] = trial_false_negatives
                    trials_by_false_positives['addb view '+file + ' --trial ' + str(reviewed_trials[i])] = trial_false_positives


                # Get the lengths of segments of present GRFs
                for i in range(len(reviewed_trials)):
                    start = -1
                    for j in range(len(reviewed_trial_missing[i])):
                        if not reviewed_trial_missing[i][j]:
                            if start == -1:
                                start = j
                        else:
                            if start != -1:
                                true_segment_lengths.append(j - start)
                                start = -1
                    if start != -1:
                        true_segment_lengths.append(len(reviewed_trial_missing[i]) - start)

                # Get the lengths of segments of predicted missing GRFs
                for i in range(len(reviewed_trials)):
                    start = -1
                    for j in range(len(estimated_missing[i])):
                        if not estimated_missing[i][j]:
                            if start == -1:
                                start = j
                        else:
                            if start != -1:
                                predicted_segment_lengths.append(j - start)
                                start = -1
                    if start != -1:
                        predicted_segment_lengths.append(len(estimated_missing[i]) - start)

            except Exception as e:
                print("Error loading: " + file)
                print(e)
                traceback.print_exc()

        # Write sorted log files
        with open('trials_by_recall.txt', 'w') as f:
            for key in sorted(trials_by_recall, key=trials_by_recall.get):
                f.write(key + ' ' + str(trials_by_recall[key]) + '\n')
        with open('trials_by_precision.txt', 'w') as f:
            for key in sorted(trials_by_precision, key=trials_by_precision.get):
                f.write(key + ' ' + str(trials_by_precision[key]) + '\n')
        with open('trials_by_f1.txt', 'w') as f:
            for key in sorted(trials_by_f1, key=trials_by_f1.get):
                f.write(key + ' ' + str(trials_by_f1[key]) + '\n')
        with open('trials_by_false_negatives.txt', 'w') as f:
            for key in sorted(trials_by_false_negatives, key=trials_by_false_negatives.get, reverse=True):
                f.write(key + ' ' + str(trials_by_false_negatives[key]) + '\n')
        with open('trials_by_false_positives.txt', 'w') as f:
            for key in sorted(trials_by_false_positives, key=trials_by_false_positives.get, reverse=True):
                f.write(key + ' ' + str(trials_by_false_positives[key]) + '\n')

        # Print statistics
        print("Num trials manually reviewed: " + str(num_trials_manually_reviewed))
        print("Num trials not manually reviewed: " + str(num_trials_not_manually_reviewed))
        print("Num missing frames: " + str(num_missing_frames))
        print("Num not missing frames: " + str(num_not_missing_frames))

        print('-----')

        total_frames = true_positives + false_positives + true_negatives + false_negatives
        print("True Positives (correct – missing GRF): " + str(true_positives) + ' - ' + str(true_positives * 100.0 / (total_frames)) + '%')
        print("False Positives (wrong - had GRF data, but we filtered frame out): " + str(false_positives) + ' - ' + str(false_positives * 100.0 / (total_frames)) + '%')
        print("True Negatives (correct - had GRF): " + str(true_negatives) + ' - ' + str(true_negatives * 100.0 / (total_frames)) + '%')
        print("False Negatives (wrong - missing GRF data, but we thought it had it): " + str(false_negatives) + ' - ' + str(false_negatives * 100.0 / (total_frames)) + '%')

        print('-----')

        if true_positives + false_positives == 0:
            print("Never predicted a missing GRF! No true positives or false positives.")
            precision = 0
        else:
            precision = true_positives / (true_positives + false_positives)

        if true_positives + false_negatives == 0:
            print("Never found a labeled missing GRF! No labeled missing frames in the dataset.")
            recall = 0
        else:
            recall = true_positives / (true_positives + false_negatives)

        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        print("Precision (of our thrown out data, how much of it should have been thrown out?): " + str(precision))
        print("Recall (how much of the bad data did we throw out?): " + str(recall))
        print("F1: " + str(f1))

        print('-----')

        if len(true_segment_lengths) == 0:
            print('No segments of usable GRFs in the dataset.')
        elif len(predicted_segment_lengths) == 0:
            print('No segments of usable GRFs predicted, so we have no stats on usable GRF segment lengths.')
        else:
            print('Predicted number of usable segments: ' + str(len(predicted_segment_lengths))+ ' compared to true value ' + str(len(true_segment_lengths)))
            print('Predicted segment median length: ' + str(np.median(predicted_segment_lengths)) + ' compared to true value ' + str(np.median(true_segment_lengths)))
            print('Predicted segment mean length: ' + str(np.mean(predicted_segment_lengths)) + ' compared to true value ' + str(np.mean(true_segment_lengths)))
            print('Predicted segment max length: ' + str(np.max(predicted_segment_lengths)) + ' compared to true value ' + str(np.max(true_segment_lengths)))
            print('Predicted segment min length: ' + str(np.min(predicted_segment_lengths)) + ' compared to true value ' + str(np.min(true_segment_lengths)))
