import argparse
import os


class AbstractCommand:
    """
    All of our different activities inherit from this class. This class defines the interface for a CLI command, so
    that it's convenient to split commands across files. It also carries shared logic for loading / saving models, etc.
    """
    def register_subcommand(self, subparsers: argparse._SubParsersAction):
        pass

    def run(self, args: argparse.Namespace) -> bool:
        pass

    def ensure_geometry(self, geometry: str):
        if geometry is None:
            # Check if the "./Geometry" folder exists, and if not, download it
            if not os.path.exists('./Geometry'):
                print('Downloading the Geometry folder from https://addbiomechanics.org/resources/Geometry.zip')
                exit_code = os.system('wget https://addbiomechanics.org/resources/Geometry.zip')
                if exit_code != 0:
                    print('ERROR: Failed to download Geometry.zip. You may need to install wget. If you are on a Mac, '
                          'try running "brew install wget"')
                    return False
                os.system('unzip ./Geometry.zip')
                os.system('rm ./Geometry.zip')
            geometry = './Geometry'
        print('Using Geometry folder: ' + geometry)
        geometry = os.path.abspath(geometry)
        if not geometry.endswith('/'):
            geometry += '/'
        return geometry