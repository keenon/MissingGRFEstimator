from cli.abstract_command import AbstractCommand
import os
import nimblephysics as nimble


class EvaluatePerformance(AbstractCommand):
    def __init__(self):
        pass

    def register_subcommand(self, subparsers):
        subparser = subparsers.add_parser('eval', help='Evaluate the performance of a missing GRF heuristic on the dataset.')
        subparser.add_argument('--dataset-home', type=str, default='../data', help='The path to the AddBiomechanics dataset.')
        subparser.add_argument('--method', type=str, default='foot-stillness', help='The method to evaluate.')

    def run(self, args):
        if 'command' in args and args.command != 'eval':
            return False
        dataset_home: str = args.dataset_home
        method: str = args.method
        print(f"Running evaluation on {dataset_home} using method {method}")

        data_dir = os.path.abspath('../data/processed')

        # Recursively list all the B3D files under data
        b3d_files = []
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith(".b3d"):
                    b3d_files.append(os.path.join(root, file))

        # Load all the B3D files, and collect statistics for each trial

        num_trials_manually_reviewed = 0
        num_trials_not_manually_reviewed = 0

        num_missing_frames = 0
        num_not_missing_frames = 0
        for file in b3d_files:
            print("Loading: " + file)
            try:
                subject: nimble.biomechanics.SubjectOnDisk = nimble.biomechanics.SubjectOnDisk(file)
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
            except Exception as e:
                print("Error loading: " + file)
                print(e)

        # Print statistics
        print("Num trials manually reviewed: " + str(num_trials_manually_reviewed))
        print("Num trials not manually reviewed: " + str(num_trials_not_manually_reviewed))
        print("Num missing frames: " + str(num_missing_frames))
        print("Num not missing frames: " + str(num_not_missing_frames))
