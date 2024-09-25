from cli.abstract_command import AbstractCommand
import os
import nimblephysics as nimble
import numpy as np


class GRFAnalyze1(AbstractCommand):
    def __init__(self):
        pass

    def register_subcommand(self, subparsers):
        subparser = subparsers.add_parser('grf-analyze1', help='Get a histogram of the linear and angular residuals before we do anything.')
        subparser.add_argument('--dataset-home', type=str, default='../data', help='The path to the AddBiomechanics dataset.')

    def run(self, args):
        if 'command' in args and args.command != 'grf-analyze1':
            return False
        dataset_home: str = args.dataset_home
        print(f"Running evaluation on {dataset_home}")

        data_dir = os.path.abspath('../data/smoothed_dynamics')

        # Recursively list all the B3D files under data
        b3d_files = []
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith(".b3d"):
                    file_path = os.path.join(root, file)
                    # if 'Carter' in file_path: # or 'Camargo' in file_path:
                    b3d_files.append(os.path.join(root, file))

        # Load all the B3D files, and collect statistics for each trial

        trial_not_missing_count = []
        trial_pass_1_linear_residuals = []
        trial_pass_2_linear_residuals = []
        trial_pass_1_angular_residuals = []
        trial_pass_2_angular_residuals = []

        for file in b3d_files:
            print("Loading: " + file)
            try:
                subject: nimble.biomechanics.SubjectOnDisk = nimble.biomechanics.SubjectOnDisk(file)

                for trial in range(subject.getNumTrials()):
                    missing_grf = subject.getMissingGRF(trial)
                    linear_residuals_pass_1 = np.array(subject.getTrialLinearResidualNorms(trial, 1))
                    angular_residuals_pass_1 = np.array(subject.getTrialAngularResidualNorms(trial, 1))

                    linear_residuals_pass_2 = []
                    angular_residuals_pass_2 = []
                    if subject.getTrialNumProcessingPasses(trial) > 2:
                        linear_residuals_pass_2 = np.array(subject.getTrialLinearResidualNorms(trial, 2))
                        angular_residuals_pass_2 = np.array(subject.getTrialAngularResidualNorms(trial, 2))

                    not_missing_count = 0.0
                    not_missing_linear_residuals_pass_1 = 0.0
                    not_missing_angular_residuals_pass_1 = 0.0
                    not_missing_linear_residuals_pass_2 = 0.0
                    not_missing_angular_residuals_pass_2 = 0.0

                    for i in range(len(missing_grf)):
                        if missing_grf[i] == nimble.biomechanics.MissingGRFReason.notMissingGRF:
                            not_missing_count += 1
                            not_missing_linear_residuals_pass_1 += linear_residuals_pass_1[i]
                            not_missing_angular_residuals_pass_1 += angular_residuals_pass_1[i]
                            if len(linear_residuals_pass_2) > 0:
                                not_missing_linear_residuals_pass_2 += linear_residuals_pass_2[i]
                                not_missing_angular_residuals_pass_2 += angular_residuals_pass_2[i]

                    if not_missing_count > 2:
                        trial_not_missing_count.append(not_missing_count)
                        trial_pass_1_linear_residuals.append(min(not_missing_linear_residuals_pass_1 / not_missing_count, 100.0))
                        trial_pass_1_angular_residuals.append(min(not_missing_angular_residuals_pass_1 / not_missing_count, 100.0))
                        if len(linear_residuals_pass_2) > 0:
                            trial_pass_2_linear_residuals.append(min(not_missing_linear_residuals_pass_2 / not_missing_count, 100.0))
                            trial_pass_2_angular_residuals.append(min(not_missing_angular_residuals_pass_2 / not_missing_count, 100.0))
            except Exception as e:
                print("Error loading: " + file)
                print(e)

        print('Num trials not missing: ' + str(len(trial_not_missing_count)))
        print("Not missing count mean: " + str(np.mean(trial_not_missing_count)))

        print("Linear residuals pass 1 mean: " + str(np.mean(trial_pass_1_linear_residuals)))
        print("Angular residuals pass 1 mean: " + str(np.mean(trial_pass_1_angular_residuals)))

        print("Linear residuals pass 2 mean: " + str(np.mean(trial_pass_2_linear_residuals)))
        print("Angular residuals pass 2 mean: " + str(np.mean(trial_pass_2_angular_residuals)))

        import matplotlib.pyplot as plt
        plt.hist(trial_pass_1_linear_residuals, bins=50, alpha=0.5, label='Pass 1')
        plt.hist(trial_pass_2_linear_residuals, bins=50, alpha=0.5, label='Pass 2')
        plt.legend()
        plt.title("Linear Residuals")
        plt.show()

        plt.hist(trial_pass_1_angular_residuals, bins=50, alpha=0.5, label='Pass 1')
        plt.hist(trial_pass_2_angular_residuals, bins=50, alpha=0.5, label='Pass 2')
        plt.legend()
        plt.title("Angular Residuals")
        plt.show()
