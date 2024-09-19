from cli.abstract_command import AbstractCommand
import os
import nimblephysics as nimble
import numpy as np


class View(AbstractCommand):
    def __init__(self):
        pass

    def register_subcommand(self, subparsers):
        subparser = subparsers.add_parser('view', help='Visualize the dataset, with given filters.')
        subparser.add_argument('--dataset-home', type=str, default='../data', help='The path to the AddBiomechanics dataset.')

    def run(self, args):
        if 'command' in args and args.command != 'analyze':
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

        # Load all the B3D files, and collect statistics for each trial

        num_trials_manually_reviewed = 0
        num_trials_not_manually_reviewed = 0

        missing_force_mags = []
        not_missing_force_mags = []

        outliers = []

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
                        reviewed_trials.append(trial)
                        reviewed_trial_missing.append([reason != nimble.biomechanics.MissingGRFReason.notMissingGRF for reason in missing_reasons])

                for i, trial in enumerate(reviewed_trials):
                    trial_len = subject.getTrialLength(trial)
                    for t in range(trial_len):
                        frame = subject.readFrames(trial, t, 1, includeSensorData=True, includeProcessingPasses=False)[
                            0]
                        forces = frame.rawForcePlateForces
                        total_force_mag = 0.0
                        for force in forces:
                            total_force_mag += np.linalg.norm(force)
                        if reviewed_trial_missing[i][t]:
                            if total_force_mag > 3000:
                                outliers.append(file+' '+str(trial)+' '+str(t)+' '+str(total_force_mag)+'N')
                            else:
                                if total_force_mag < 50:
                                    missing_force_mags.append(total_force_mag)
                        else:
                            if total_force_mag < 50:
                                not_missing_force_mags.append(total_force_mag)

            except Exception as e:
                print("Error loading: " + file)
                print(e)

        print('Outliers: '+str(len(outliers)))
        with open('outliers.txt', 'w') as f:
            for outlier in outliers:
                f.write(outlier+'\n')

        print('Number of manually reviewed trials: ' + str(num_trials_manually_reviewed))
        print('Number of missing frames: ' + str(len(missing_force_mags)))
        print('Mean missing force magnitude: ' + str(np.mean(missing_force_mags)))
        print('Median missing force magnitude: ' + str(np.median(missing_force_mags)))

        print('Number of not missing frames: ' + str(len(not_missing_force_mags)))
        print('Mean not missing force magnitude: ' + str(np.mean(not_missing_force_mags)))
        print('Median not missing force magnitude: ' + str(np.median(not_missing_force_mags)))


        # Plot a histogram of the total force magnitudes
        import matplotlib.pyplot as plt
        plt.hist(missing_force_mags, bins=50, alpha=1.0, label='Missing')
        plt.title('Missing Force Magnitudes')
        plt.show()

        plt.hist(not_missing_force_mags, bins=50, alpha=1.0, label='Not Missing')
        plt.title('Not Missing Force Magnitudes')
        plt.show()
