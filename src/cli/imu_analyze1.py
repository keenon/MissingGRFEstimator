from cli.abstract_command import AbstractCommand
import os
import nimblephysics as nimble
import numpy as np


class IMUAnalyze1(AbstractCommand):
    def __init__(self):
        pass

    def register_subcommand(self, subparsers):
        subparser = subparsers.add_parser('imu-analyze1', help='Analyze the raw IMU data.')
        subparser.add_argument('--dataset-home', type=str, default='../data', help='The path to the AddBiomechanics dataset.')

    def run(self, args):
        if 'command' in args and args.command != 'imu-analyze1':
            return False
        dataset_home: str = args.dataset_home
        print(f"Running evaluation on {dataset_home}")

        data_dir = os.path.abspath('../data/smoothed/protected/us-west-2:e013a4d2-683d-48b9-bfe5-83a0305caf87/data/Camargo2021_Formatted_No_Arm/AB18_split4')

        # Recursively list all the B3D files under data
        b3d_files = []
        csv_files = []
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith(".b3d"):
                    b3d_files.append(os.path.join(root, file))
                if file.endswith(".csv"):
                    csv_files.append(os.path.join(root, file))

        # Load a CSV file
        if len(csv_files) > 0:
            print("Loading: " + csv_files[0])
            data = np.genfromtxt(csv_files[0], delimiter=',', skip_header=1)
            print(data.shape)
            # Plot the data
            import matplotlib.pyplot as plt
            time = data[:, 0]
            plt.plot(time, data[:, 1], label='foot X')
            plt.plot(time, data[:, 2], label='foot Y')
            plt.plot(time, data[:, 3], label='foot Z')
            plt.legend()
            plt.title(csv_files[0])
            plt.show()

            plt.plot(time, data[:, 7], label='shank X')
            plt.plot(time, data[:, 8], label='shank Y')
            plt.plot(time, data[:, 9], label='shank Z')
            plt.legend()
            plt.title(csv_files[0])
            plt.show()

            plt.plot(time, data[:, 13], label='thigh X')
            plt.plot(time, data[:, 14], label='thigh Y')
            plt.plot(time, data[:, 15], label='thigh Z')
            plt.legend()
            plt.title(csv_files[0])
            plt.show()

            plt.plot(time, data[:, 19], label='root X')
            plt.plot(time, data[:, 20], label='root Y')
            plt.plot(time, data[:, 21], label='root Z')
            plt.legend()
            plt.title(csv_files[0])
            plt.show()
