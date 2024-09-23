from cli.abstract_command import AbstractCommand
import os
import nimblephysics as nimble
import numpy as np
import traceback
import random
import matplotlib.pyplot as plt


class GRFAnalyze7(AbstractCommand):
    def __init__(self):
        pass

    def register_subcommand(self, subparsers):
        subparser = subparsers.add_parser('grf-analyze7', help='Compare the linear residuals to our analytical computation.')
        subparser.add_argument('--dataset-home', type=str, default='../data', help='The path to the AddBiomechanics dataset.')

    def run(self, args):
        if 'command' in args and args.command != 'grf-analyze7':
            return False
        dataset_home: str = args.dataset_home
        print(f"Running evaluation on {dataset_home}")

        data_dir = os.path.abspath('../data/smoothed')

        # Recursively list all the B3D files under data
        b3d_files = []
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith(".b3d"):
                    file_path = os.path.join(root, file)
                    # if 'Santos' in file_path:
                    #     continue
                    b3d_files.append(os.path.join(root, file))
        random.seed(42)
        random.shuffle(b3d_files)
        b3d_files = b3d_files[:10]

        # Load all the B3D files, and collect statistics for each trial

        trial_angular_residual_sums = []
        trial_angular_residual_abs_sums = []
        trial_translations = []

        for file in b3d_files:
            print("Loading: " + file)
            try:
                subject: nimble.biomechanics.SubjectOnDisk = nimble.biomechanics.SubjectOnDisk(file)
                subject.loadAllFrames(doNotStandardizeForcePlateData=True)

                skel: nimble.dynamics.Skeleton = subject.readSkel(1, ignoreGeometry=True)

                trial_protos = subject.getHeaderProto().getTrials()

                force_body_indices = [skel.getBodyNode(body_name).getIndexInSkeleton() for body_name in subject.getGroundForceBodies()]
                foot_body_names_recovered = [skel.getBodyNode(i).getName() for i in force_body_indices]
                assert foot_body_names_recovered == subject.getGroundForceBodies()
                residual_helper: nimble.biomechanics.ResidualForceHelper = nimble.biomechanics.ResidualForceHelper(skel, force_body_indices)

                for trial in range(subject.getNumTrials()):
                    dt = subject.getTrialTimestep(trial)

                    missing_grf = subject.getMissingGRF(trial)
                    linear_residuals = np.array(subject.getTrialLinearResidualNorms(trial, 1))
                    angular_residuals = np.array(subject.getTrialAngularResidualNorms(trial, 1))

                    trial_proto = trial_protos[trial]
                    smoothed_pass = trial_proto.getPasses()[1]
                    poses = smoothed_pass.getPoses()
                    vels = smoothed_pass.getVels()
                    accs = smoothed_pass.getAccs()
                    com_poses = smoothed_pass.getComPoses()
                    raw_forces = smoothed_pass.getGroundBodyWrenches()

                    angular_residual_sum = np.zeros(3)
                    absolute_residual_sum = np.zeros(3)
                    trial_translation = np.zeros(3)
                    all_translation_distances = []
                    total_frames = 0

                    for t in range(len(missing_grf)):
                        if missing_grf[t] == nimble.biomechanics.MissingGRFReason.notMissingGRF and t > 0 and t < len(missing_grf) - 1:
                            tau = residual_helper.calculateInverseDynamics(poses[:, t], vels[:, t], accs[:, t], raw_forces[:, t])
                            angular_residual = tau[0:3]

                            original_com = com_poses[:, t]
                            target_com = residual_helper.calculateComToCenterAngularResiduals(poses[:, t], vels[:, t], accs[:, t], raw_forces[:, t])
                            translation = target_com - original_com
                            all_translation_distances.append(translation)
                            trial_translation += translation

                            angular_residual_sum += angular_residual
                            absolute_residual_sum += np.abs(angular_residual)
                            total_frames += 1
                        else:
                            all_translation_distances.append(np.zeros(3))

                    if total_frames > 0:
                        angular_residual_sum /= total_frames
                        absolute_residual_sum /= total_frames
                        trial_translation /= total_frames

                        print("Trial: " + str(trial)+' translation '+str(np.linalg.norm(trial_translation))+'m')
                        if np.linalg.norm(trial_translation) > 0.05 or True:
                            all_translation_distances = np.array(all_translation_distances)

                            plt.plot(all_translation_distances[:, 0], label='x')
                            plt.plot(all_translation_distances[:, 1], label='y')
                            plt.plot(all_translation_distances[:, 2], label='z')
                            plt.legend()
                            plt.title("Translation distances")
                            plt.show()

                        trial_translations.append(np.linalg.norm(trial_translation))
                        trial_angular_residual_sums.append(np.linalg.norm(angular_residual_sum))
                        trial_angular_residual_abs_sums.append(np.linalg.norm(absolute_residual_sum))

            except Exception as e:
                print("Error loading: " + file)
                print(e)
                traceback.print_exc()

        print("Trial translations: " + str(np.mean(trial_translations)))
        print("Trial angular residuals sum: " + str(np.mean(trial_angular_residual_sums)))
        print("Trial absolute angular residuals sum: " + str(np.mean(trial_angular_residual_abs_sums)))

        plt.hist(trial_translations, bins=20)
        plt.title("Translations")
        plt.show()

        plt.hist(trial_angular_residual_sums, bins=20)
        plt.title("Angular Residuals Sum (net moment over trial)")
        plt.show()
