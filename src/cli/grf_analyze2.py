from cli.abstract_command import AbstractCommand
import os
import nimblephysics as nimble
import numpy as np
import traceback


class GRFAnalyze2(AbstractCommand):
    def __init__(self):
        pass

    def register_subcommand(self, subparsers):
        subparser = subparsers.add_parser('grf-analyze2', help='Compare the linear residuals to our analytical computation.')
        subparser.add_argument('--dataset-home', type=str, default='../data', help='The path to the AddBiomechanics dataset.')

    def run(self, args):
        if 'command' in args and args.command != 'grf-analyze2':
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
                    if 'Santos' in file_path:
                        continue
                    b3d_files.append(os.path.join(root, file))

        # Load all the B3D files, and collect statistics for each trial

        trial_linear_residuals = []

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
                    raw_forces = smoothed_pass.getGroundBodyWrenches()
                    com_accs = smoothed_pass.getComAccs()

                    assert poses.shape == vels.shape
                    assert poses.shape == accs.shape

                    # smoothed_pass.computeKinematicValues(skel, dt, poses)
                    # updated_vels = smoothed_pass.getVels()
                    # updated_accs = smoothed_pass.getAccs()
                    # updated_com_accs = smoothed_pass.getComAccs()
                    #
                    # assert np.allclose(vels, updated_vels)
                    # assert np.allclose(accs, updated_accs)
                    # assert np.allclose(com_accs, updated_com_accs)

                    cop_torque_force_in_world = smoothed_pass.getGroundBodyCopTorqueForce()
                    num_force_plates = int(cop_torque_force_in_world.shape[0] / 9)
                    num_contact_bodies = len(force_body_indices)

                    for t in range(len(missing_grf)):
                        if missing_grf[t] == nimble.biomechanics.MissingGRFReason.notMissingGRF:
                            # Check the baseline COM acc
                            skel.setPositions(poses[:, t])
                            skel.setVelocities(vels[:, t])
                            skel.setAccelerations(accs[:, t])
                            com_world_acc = skel.getCOMLinearAcceleration() - skel.getGravity()
                            given_com_world_acc = com_accs[:, t]
                            error = np.linalg.norm(com_world_acc - given_com_world_acc)
                            if error > 1e-6:
                                print("Error: " + str(error))
                                print("COM acc: " + str(com_world_acc))
                                print("Given COM acc: " + str(given_com_world_acc))
                                print("Trial: " + str(trial))
                                print("Frame: " + str(t))
                                print("File: " + file)
                                continue

                            total_force = np.zeros(3)
                            for j in range(num_force_plates):
                                total_force += cop_torque_force_in_world[j * 9 + 6:j * 9 + 9, t]

                            # total_foot_force = np.zeros(3)
                            # for j in range(num_contact_bodies):
                            #     total_foot_force += raw_forces[j * 6 + 3:j * 6 + 6, t]
                            # error = np.linalg.norm(total_force - total_foot_force)
                            # if error > 1e-6:
                            #     print("Error: " + str(error))
                            #     print("Total force: " + str(total_force))
                            #     print("Total foot force: " + str(total_foot_force))
                            #     print("Trial: " + str(trial))
                            #     print("Frame: " + str(t))
                            #     print("File: " + file)
                            #     continue

                            # Check the baseline inverse dynamics
                            plain_taus = skel.getInverseDynamics(accs[:, t])
                            helper_plain_tau = residual_helper.calculateInverseDynamics(poses[:, t], vels[:, t], accs[:, t], np.zeros_like(raw_forces[:, t]))
                            error = np.linalg.norm(plain_taus - helper_plain_tau)
                            if error > 1e-6:
                                print("Error: " + str(error))
                                print("Plain Tau: " + str(plain_taus))
                                print("Helper plain tau: " + str(helper_plain_tau))
                                print("Trial: " + str(trial))
                                print("Frame: " + str(t))
                                print("File: " + file)
                                continue

                            actual_residual = linear_residuals[t]
                            tau = residual_helper.calculateInverseDynamics(poses[:, t], vels[:, t], accs[:, t], raw_forces[:, t])
                            implied_residual_from_tau = np.linalg.norm(tau[3:6])
                            error = abs(implied_residual_from_tau - actual_residual)
                            if error > 0.01:
                                print("Error from tau: " + str(error))
                                print("Implied residual from tau: " + str(tau[3:6]))
                                print("Implied residual from implied_force - force: " + str(implied_force - total_force))
                                print("Implied force: " + str(implied_force))
                                print("COM acc: " + str(com_accs[:, t]))
                                print("Mass: " + str(skel.getMass()))
                                print("Tau: " + str(tau[3:6]))
                                print("Trial: " + str(trial))
                                print("Frame: " + str(t))
                                print("File: " + file)
                                continue

                            implied_force = com_accs[:, t] * skel.getMass()
                            implied_residual = np.linalg.norm(total_force - implied_force)
                            error = abs(implied_residual - actual_residual)
                            if error > 0.01:
                                print("Error: " + str(error))
                                print("Implied residual: " + str(implied_residual))
                                print("Actual residual: " + str(actual_residual))
                                print("Total force: " + str(total_force))
                                print("Implied force: " + str(implied_force))
                                print("COM acc: " + str(com_accs[:, t]))
                                print("Mass: " + str(skel.getMass()))
                                print("Trial: " + str(trial))
                                print("Frame: " + str(t))
                                print("File: " + file)
                                continue

                            implied_angular_residual_vec = tau[0:3]
                            implied_angular_residual = np.linalg.norm(implied_angular_residual_vec)
                            actual_angular_residual = angular_residuals[t]
                            angular_error = abs(implied_angular_residual - actual_angular_residual)
                            if angular_error > 0.01:
                                print("Angular residual Error: " + str(angular_error))
                                print("Implied angular residual: " + str(implied_angular_residual_vec))
                                print("Implied angular residual norm: " + str(implied_angular_residual))
                                print("Actual angular residual norm: " + str(actual_angular_residual))
                                print("Trial: " + str(trial))
                                print("Frame: " + str(t))
                                print("File: " + file)
                                continue

                            # spatial_residual_vec = residual_helper.calculateCOMAngularResidual(poses[:, t], vels[:, t], accs[:, t], raw_forces[:, t])
                            # spatial_residual = np.linalg.norm(spatial_residual_vec)
                            # error = abs(spatial_residual - actual_angular_residual)
                            # if error > 1.0:
                            #     print("Spatial Error: " + str(error))
                            #     print("Implied spatial residual: " + str(spatial_residual_vec))
                            #     print("Implied spatial residual norm: " + str(spatial_residual))
                            #     print("Actual spatial residual norm: " + str(actual_angular_residual))
                            #     print("Trial: " + str(trial))
                            #     print("Frame: " + str(t))
                            #     print("File: " + file)
                            #     continue

            except Exception as e:
                print("Error loading: " + file)
                print(e)
                traceback.print_exc()

        print("Linear residuals mean: " + str(np.mean(trial_linear_residuals)))

        import matplotlib.pyplot as plt
        plt.hist(trial_linear_residuals, bins=20)
        plt.title("Linear Residuals")
        plt.show()
