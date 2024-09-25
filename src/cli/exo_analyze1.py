import time
import random
from cli.abstract_command import AbstractCommand
import os
import nimblephysics as nimble
import numpy as np
import traceback


class ExoAnalyze1(AbstractCommand):
    def __init__(self):
        pass

    def register_subcommand(self, subparsers):
        subparser = subparsers.add_parser('exo-analyze1', help='Try to draw some simple conclusions for exoskeletons based on the data we have so far.')
        subparser.add_argument('--dataset-home', type=str, default='../data', help='The path to the AddBiomechanics dataset.')

    def run(self, args):
        if 'command' in args and args.command != 'exo-analyze1':
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
                    # if 'Carter' in file_path or 'Camargo' in file_path:
                    b3d_files.append(file_path)

        # Load all the B3D files, and collect statistics for each trial

        dofs = [
            'knee_angle_l',
            'knee_angle_r',
            'hip_flexion_l',
            'hip_flexion_r'
        ]
        dof_taus = {dof: [] for dof in dofs}
        dof_vels = {dof: [] for dof in dofs}
        dof_pwrs = {dof: [] for dof in dofs}

        for file in b3d_files:
            print("Loading: " + file)
            try:
                subject: nimble.biomechanics.SubjectOnDisk = nimble.biomechanics.SubjectOnDisk(file)

                skel: nimble.dynamics.Skeleton = subject.readSkel(0, ignoreGeometry=True)
                for d in range(skel.getNumDofs()):
                    print(skel.getDofByIndex(d).getName())
                dof_indices = [skel.getDof(dof).getIndexInSkeleton() for dof in dofs]

                for trial in range(subject.getNumTrials()):
                    if subject.getTrialNumProcessingPasses(trial) < 3:
                        continue
                    frames = subject.readFrames(trial, 0, subject.getTrialLength(trial), includeSensorData=False, includeProcessingPasses=True)
                    for t in range(len(frames)):
                        frame = frames[t]
                        if frame.missingGRFReason != nimble.biomechanics.MissingGRFReason.notMissingGRF:
                            continue
                        tau = frame.processingPasses[2].tau
                        vel = frame.processingPasses[2].vel

                        for dof, index in zip(dofs, dof_indices):
                            dof_taus[dof].append(tau[index])
                            dof_vels[dof].append(vel[index])
                            dof_pwrs[dof].append(tau[index] * vel[index])
            except Exception as e:
                print("Error loading: " + file)
                print(e)
                traceback.print_exc()

        import matplotlib.pyplot as plt

        for dof in dofs:
            plt.figure()
            plt.hist(dof_taus[dof], bins=100, alpha=0.5, label='Tau')
            plt.hist(dof_vels[dof], bins=100, alpha=0.5, label='Vel')
            plt.hist(dof_pwrs[dof], bins=100, alpha=0.5, label='Pwr')
            plt.legend()
            plt.title(dof)
            plt.show()
