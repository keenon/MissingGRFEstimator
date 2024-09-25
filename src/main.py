import argparse

import nimblephysics_libs.biomechanics

from cli.abstract_command import AbstractCommand
from cli.evaluate_performance import EvaluatePerformance
from cli.analyze import Analyze
from cli.analyze2 import Analyze2
from cli.analyze3 import Analyze3
from cli.analyze4 import Analyze4
from cli.analyze5 import Analyze5
from cli.analyze6 import Analyze6
from cli.analyze7 import Analyze7
from cli.analyze8 import Analyze8
from cli.grf_analyze1 import GRFAnalyze1
from cli.grf_analyze2 import GRFAnalyze2
from cli.grf_analyze3 import GRFAnalyze3
from cli.grf_analyze4 import GRFAnalyze4
from cli.grf_analyze5 import GRFAnalyze5
from cli.grf_analyze6 import GRFAnalyze6
from cli.grf_analyze7 import GRFAnalyze7
from cli.grf_analyze8 import GRFAnalyze8
from cli.view_markers import ViewMarkers
from cli.exo_analyze1 import ExoAnalyze1
from cli.exo_analyze2 import ExoAnalyze2
from cli.imu_analyze1 import IMUAnalyze1
from cli.imu_analyze2 import IMUAnalyze2
from cli.imu_analyze3 import IMUAnalyze3
from cli.imu_analyze4 import IMUAnalyze4
from cli.imu_analyze5 import IMUAnalyze5
from cli.imu_analyze6 import IMUAnalyze6


def main():
    commands = [
        EvaluatePerformance(),
        Analyze(),
        Analyze2(),
        Analyze3(),
        Analyze4(),
        Analyze5(),
        Analyze6(),
        Analyze7(),
        Analyze8(),
        ViewMarkers(),
        GRFAnalyze1(),
        GRFAnalyze2(),
        GRFAnalyze3(),
        GRFAnalyze4(),
        GRFAnalyze5(),
        GRFAnalyze6(),
        GRFAnalyze7(),
        GRFAnalyze8(),
        ExoAnalyze1(),
        ExoAnalyze2(),
        IMUAnalyze1(),
        IMUAnalyze2(),
        IMUAnalyze3(),
        IMUAnalyze4(),
        IMUAnalyze5(),
        IMUAnalyze6()
    ]

    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(
        description='InferBiomechanics Command Line Interface')

    # Split up by command
    subparsers = parser.add_subparsers(dest="command")

    # Add a parser for each command
    for command in commands:
        command.register_subcommand(subparsers)

    # Parse the arguments
    args = parser.parse_args()

    for command in commands:
        if command.run(args):
            return


if __name__ == '__main__':
    # logpath = "log"
    # # Create and configure logger
    # logging.basicConfig(filename=logpath,
    #                     format='%(asctime)s %(message)s',
    #                     filemode='a')
    #
    # # Creating an object
    # logger = logging.getLogger()
    # logger.addHandler(logging.StreamHandler())
    # # Setting the threshold of logger to INFO
    # logger.setLevel(logging.INFO)
    main()
