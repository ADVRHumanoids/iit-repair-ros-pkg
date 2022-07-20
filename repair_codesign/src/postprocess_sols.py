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
from codesign_pyutils.misc_definitions import get_design_map

import matplotlib.pyplot as plt

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

def remove_lr(var_name):

    var_name.replace() 
def extract_q_design(input_data):

    design_var_map = get_design_map()

    design_indeces = [design_var_map["mount_h"],\
        design_var_map["should_wl"],\
        design_var_map["should_roll_l"],\
        design_var_map["wrist_off_l"]]

    n_samples = len(input_data)

    design_data = np.zeros((len(design_indeces), n_samples))

    for i in range(n_samples):

        design_data[:, i] = input_data[i][design_indeces, 0]

    return design_data

def main(args):

    try:

        sliding_wrist_command = "is_sliding_wrist:=" + "true"

        # print(sliding_wrist_command)
        xacro_gen = subprocess.check_call(["xacro",\
                                        xacro_full_path, \
                                        sliding_wrist_command, \
                                        "-o", 
                                        urdf_full_path])

    except:

        print('Failed to generate URDF.')

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
    opt_costs = np.zeros((n_opt_sol, 1)).flatten()

    for i in range(n_opt_sol):

        opt_full_q[i] = sol_loader.opt_data[i]["q"]
        opt_full_q_dot[i] = sol_loader.opt_data[i]["q_dot"]
        opt_costs[i] = sol_loader.opt_data[i]["opt_cost"]

    opt_q_design = extract_q_design(opt_full_q)

    # print(opt_q_design)
    # exit()
    n_d_variables = np.shape(opt_q_design)[0]

    design_var_map = get_design_map()
    design_var_names = list(design_var_map.keys())
    
    opt_index = np.where(opt_costs == min(opt_costs))


    # scatter plots
    for i in range(n_d_variables):
    
        plt.figure()
        plt.scatter(opt_costs, opt_q_design[i, :], label=r"", marker="o", s=50 )
        plt.legend(loc="upper left")
        plt.xlabel(r"opt_cost")
        plt.ylabel(design_var_names[i])
        plt.title(design_var_names[i], fontdict=None, loc='center')
        plt.grid()
    
    # histograms
    for i in range(n_d_variables):
    
        plt.figure()
        plt.hist(opt_q_design[i, :], bins = 100)
        plt.scatter(opt_q_design[i, opt_index], 0, label=r"", marker="x", s=200, color="orange", linewidth=3)
        plt.legend(loc="upper left")
        plt.xlabel(r"")
        plt.ylabel(design_var_names[i])
        plt.title(design_var_names[i], fontdict=None, loc='center')
        plt.grid()

    plt.show() # show the plots



if __name__ == '__main__':

    # adding script arguments
    parser = argparse.ArgumentParser(
        description='just a simple test file for RePAIR co-design')
    parser.add_argument('--resample_sol', '-rs', type=str2bool,\
                        help = 'whether to resample the obtained solution before replaying it', default = False)

    args = parser.parse_args()

    main(args)