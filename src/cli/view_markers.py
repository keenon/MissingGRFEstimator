from cli.abstract_command import AbstractCommand
import os
import nimblephysics as nimble
import numpy as np
import json
from typing import List, Optional


class ViewMarkers(AbstractCommand):
    def __init__(self):
        pass

    def register_subcommand(self, subparsers):
        subparser = subparsers.add_parser('view-markers', help='Visualize the dataset, with given filters.')
        subparser.add_argument('--dataset-home', type=str, default='../data', help='The path to the AddBiomechanics dataset.')
        subparser.add_argument('--foot-marker-file', type=str, default='../foot_marker_file.json', help='The JSON file containing the locations of the markers on the foot.')

    def run(self, args):
        if 'command' in args and args.command != 'view-markers':
            return False
        dataset_home: str = args.dataset_home
        foot_marker_file: str = args.foot_marker_file

        print(f"Running evaluation on {dataset_home}")

        data_dir = os.path.abspath('../data/processed')

        # Recursively list all the B3D files under data
        b3d_files = []
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith(".b3d"):
                    b3d_files.append(os.path.join(root, file))

        subject_on_disk = nimble.biomechanics.SubjectOnDisk(b3d_files[0])
        osim = subject_on_disk.readOpenSimFile(processingPass=0, geometryFolder=os.path.abspath('../Geometry')+'/')

        with open(foot_marker_file) as f:
            foot_marker_data = json.load(f)

        gui = nimble.NimbleGUI()
        gui.serve(8080)
        gui.nativeAPI().renderSkeleton(osim.skeleton)

        for marker_name in foot_marker_data:
            marker = foot_marker_data[marker_name]
            mesh_patterns: List[str] = marker['mesh_patterns']
            mesh_name: Optional[str] = None
            for mesh in osim.meshMap:
                if any([pattern in mesh for pattern in mesh_patterns]):
                    mesh_name = mesh
                    break
            if mesh_name is None:
                print(f"Could not find mesh for marker {marker_name}")
                continue
            offset: np.ndarray = np.array(marker['offset'])
            body_name: str = osim.meshMap[mesh_name][0]
            relative_T: nimble.math.Isometry3 = osim.meshMap[mesh_name][1]
            body: nimble.dynamics.BodyNode = osim.skeleton.getBodyNode(body_name)
            body_offset: np.ndarray = relative_T.multiply(offset)
            world_pos: np.ndarray = body.getWorldTransform().multiply(body_offset)
            gui.nativeAPI().createBox(marker_name, [0.01, 0.01, 0.01], world_pos, [0, 0, 0], [1, 0, 0, 1])
            gui.nativeAPI().setObjectTooltip(marker_name, marker_name)

        gui.blockWhileServing()
