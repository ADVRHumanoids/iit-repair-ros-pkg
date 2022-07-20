#!/usr/bin/env python3

from horizon.utils.resampler_trajectory import resampler
from horizon.ros.replay_trajectory import *

import os, argparse
from os.path import exists

import subprocess

import rospkg

from codesign_pyutils.ros_utils import ReplaySol
from codesign_pyutils.miscell_utils import str2bool,\
                                        get_min_cost_index
            
from codesign_pyutils.tasks import FlippingTaskGen
from codesign_pyutils.load_utils import LoadSols

# useful paths
rospackage = rospkg.RosPack() # Only for taking the path to the leg package

urdfs_path = rospackage.get_path("repair_urdf") + "/urdf"
urdf_name = "repair_full"
urdf_full_path = urdfs_path + "/" + urdf_name + ".urdf"
xacro_full_path = urdfs_path + "/" + urdf_name + ".urdf.xacro"

codesign_path = rospackage.get_path("repair_codesign")

results_path = codesign_path + "/test_results"

replay_folder_name = "replay_directory" 
replay_base_path = results_path  + "/" + replay_folder_name

# resample solutions before replaying
refinement_scale = 10

def extract_q_design(input_data):

    nq = np.shape(input_data[0])[0]
    n_samples = len(input_data)

    design_data = np.zeros((nq, n_samples))

    design_indeces = 
    return design_data

def main(args):

    # only used to parse urdf
    dummy_task = FlippingTaskGen()

    dummy_task.add_in_place_flip_task(0)

    # initialize problem
    dummy_task.init_prb(urdf_full_path)

    sol_loader = LoadSols(replay_base_path)
    
    n_opt_sol = len(sol_loader.opt_data)

    opt_costs = [1e6] * n_opt_sol

    opt_full_q = [None] * n_opt_sol
    opt_full_q_dot = [None] * n_opt_sol

    for i in range(n_opt_sol):

        opt_full_q[i] = sol_loader.opt_data[i]["q"]
        opt_full_q_dot[i] = sol_loader.opt_data[i]["q_dot"]
    
    opt_q_design = 
    opt_q_dot_design = 
if __name__ == '__main__':

    # adding script arguments
    parser = argparse.ArgumentParser(
        description='just a simple test file for RePAIR co-design')
    parser.add_argument('--resample_sol', '-rs', type=str2bool,\
                        help = 'whether to resample the obtained solution before replaying it', default = False)

    args = parser.parse_args()

    main(args)