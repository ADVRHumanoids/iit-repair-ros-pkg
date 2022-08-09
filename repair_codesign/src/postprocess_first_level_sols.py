#!/usr/bin/env python3

import argparse

import subprocess

import rospkg

from codesign_pyutils.miscell_utils import Clusterer, str2bool
            
from codesign_pyutils.tasks import TaskGen
from codesign_pyutils.load_utils import LoadSols
from codesign_pyutils.misc_definitions import get_design_map

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

import numpy as np

from codesign_pyutils.miscell_utils import extract_q_design, compute_man_measure, scatter3Dcodesign, select_best_sols

from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering

from itertools import cycle, islice

import time

from codesign_pyutils.miscell_utils import Clusterer

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

    for i in range(n_opt_sol):

        opt_full_q[i] = sol_loader.opt_data[i]["q"]
        opt_full_q_dot[i] = sol_loader.opt_data[i]["q_dot"]
        opt_costs[i] = sol_loader.opt_data[i]["opt_cost"][0][0] # [0][0] because MatStorer loads matrices by default

    opt_q_design = extract_q_design(opt_full_q)

    n_d_variables = np.shape(opt_q_design)[0]

    design_var_map = get_design_map()
    design_var_names = list(design_var_map.keys())
    
    opt_index = np.argwhere(np.array(opt_costs) == min(np.array(opt_costs)))[0][0]
    opt_sol_index = sol_loader.opt_data[opt_index]["solution_index"][0][0] # [0][0] because MatStorer loads matrices by default

    n_int = len(opt_full_q_dot[0][0, :]) # getting number of intervals of a single optimization task
    man_measure = compute_man_measure(opt_costs, n_int) # scaling opt costs to make them more interpretable

    # # scatter plots
    # for i in range(n_d_variables):
    
    #     plt.figure()
    #     plt.scatter(man_measure, opt_q_design[i, :], label=r"", marker="o", s=50 )
    #     plt.legend(loc="upper left")
    #     plt.xlabel(r"rad/s")
    #     plt.ylabel(design_var_names[i])
    #     plt.title(design_var_names[i], fontdict=None, loc='center')
    #     plt.grid()
    
    # 1D histograms (w.r.t. co-design variables)
    # for i in range(n_d_variables):
    
    #     plt.figure()
    #     plt.hist(opt_q_design[i, :], bins = 100)
    #     plt.scatter(opt_q_design[i, opt_index], 0, label=r"", marker="x", s=200, color="orange", linewidth=3)
    #     plt.legend(loc="upper left")
    #     plt.xlabel(r"")
    #     plt.ylabel(r"N. sol")
    #     plt.title(design_var_names[i], fontdict=None, loc='center')
    #     plt.grid()

    #1D histogram (w.r.t. perfomance index) 
    plt.figure()
    plt.hist(man_measure, bins = 200)
    plt.legend(loc="upper left")
    plt.xlabel(r"rad/s")
    plt.ylabel(r"N. sol")
    plt.title(r"Performance index histogram", fontdict=None, loc='center')
    plt.grid()

    # 3D scatterplots
    opt_q_design_selections, opt_costs_sorted = select_best_sols(100, opt_costs, opt_q_design)
    scatter3Dcodesign(opt_costs, opt_costs_sorted, opt_q_design_selections,  n_int)

    clusterer = Clusterer(opt_q_design.T, opt_costs, n_int, n_clusters = 40)

    clusterer.clusterize()

    # algo_names = clusterer.get_algo_names()

    # for i in range(clusterer.get_n_clust()):

    #     print(clusterer.get_cluster_data(i))
    #     print("\n\n")
    clusterer.create_cluster_plot(show_clusters_sep = True, 
                                    show_cluster_costs = True)
    clusterer.show_plots()
    

if __name__ == '__main__':

    # adding script arguments
    parser = argparse.ArgumentParser(
        description='just a simple test file for RePAIR co-design')
    parser.add_argument('--resample_sol', '-rs', type=str2bool,\
                        help = 'whether to resample the obtained solution before replaying it', default = False)

    args = parser.parse_args()

    main(args)