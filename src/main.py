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
from cli.view_markers import ViewMarkers
import nimblephysics as nimble
import logging


def main():
    commands = [EvaluatePerformance(), Analyze(), Analyze2(), Analyze3(), Analyze4(), Analyze5(), Analyze6(), Analyze7(), Analyze8(), ViewMarkers()]

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
