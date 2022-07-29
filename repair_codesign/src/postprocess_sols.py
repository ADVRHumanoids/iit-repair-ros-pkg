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
            
from codesign_pyutils.tasks import TaskGen
from codesign_pyutils.load_utils import LoadSols
from codesign_pyutils.misc_definitions import get_design_map

import matplotlib.pyplot as plt

from mpl_toolkits import mplot3d
import numpy as np
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

def extract_q_design(input_data):

    design_var_map = get_design_map()

    design_indeces = [design_var_map["mount_h"],\
        design_var_map["should_wl"],\
        design_var_map["should_roll_l"],\
        design_var_map["wrist_off_l"]]

    n_samples = len(input_data)

    design_data = np.zeros((len(design_indeces), n_samples))

    for i in range(n_samples):

        design_data[:, i] = input_data[i][design_indeces, 0] # design variables are constant over the nodes (index 0 is sufficient)

    return design_data

def compute_man_measure(opt_costs, n_int):

    man_measure = np.zeros((len(opt_costs), 1))

    for i in range(len(opt_costs)): 

        man_measure[i] = np.sqrt(opt_costs[i] / n_int) # --> discretized root mean squared joint velocities over the opt interval 

    return man_measure

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
    dummy_task = TaskGen()

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
    n_int = len(opt_full_q_dot[0][0, :]) # getting number of intervals 
    man_measure = compute_man_measure(opt_costs, n_int) # scaling opt costs to make them more interpretable

    # scatter plots
    for i in range(n_d_variables):
    
        plt.figure()
        plt.scatter(man_measure, opt_q_design[i, :], label=r"", marker="o", s=50 )
        plt.legend(loc="upper left")
        plt.xlabel(r"rad/s")
        plt.ylabel(design_var_names[i])
        plt.title(design_var_names[i], fontdict=None, loc='center')
        plt.grid()
    
    # 1D histograms (w.r.t. co-design variables)
    for i in range(n_d_variables):
    
        plt.figure()
        plt.hist(opt_q_design[i, :], bins = 100)
        plt.scatter(opt_q_design[i, opt_index], 0, label=r"", marker="x", s=200, color="orange", linewidth=3)
        plt.legend(loc="upper left")
        plt.xlabel(r"")
        plt.ylabel(r"N. sol")
        plt.title(design_var_names[i], fontdict=None, loc='center')
        plt.grid()

    # 1D histogram (w.r.t. perfomance index) 
    plt.figure()
    plt.hist(man_measure, bins = 200)
    plt.legend(loc="upper left")
    plt.xlabel(r"rad/s")
    plt.ylabel(r"N. sol")
    plt.title(r"Cost histogram", fontdict=None, loc='center')
    plt.grid()

    # 3D scatterplot of mounting height, shoulder width and roll mounting angle + colormap on the performance index
    fig = plt.figure()
    ax = plt.axes(projection ="3d")
    ax.grid(b = True, color ='grey',
        linestyle ='-.', linewidth = 0.3,
        alpha = 0.2)
    my_cmap = plt.get_cmap('jet_r')

    sctt = ax.scatter3D(opt_q_design[0, :],\
                        opt_q_design[1, :],\
                        opt_q_design[2, :],\
                        alpha = 0.8,
                        c = man_measure.flatten(),
                        cmap = my_cmap,
                        marker ='o')
    plt.title("Co-design variables scatter plot")
    ax.set_xlabel('mount. height', fontweight ='bold')
    ax.set_ylabel('should. width', fontweight ='bold')
    ax.set_zlabel('mount. roll angle', fontweight ='bold')
    fig.colorbar(sctt, ax = ax, shrink = 0.5, aspect = 20, label='performance index')

    # clustering test

    

    plt.show() # show all plots


if __name__ == '__main__':

    # adding script arguments
    parser = argparse.ArgumentParser(
        description='just a simple test file for RePAIR co-design')
    parser.add_argument('--resample_sol', '-rs', type=str2bool,\
                        help = 'whether to resample the obtained solution before replaying it', default = False)

    args = parser.parse_args()

    main(args)