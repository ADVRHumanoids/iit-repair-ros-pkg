#!/usr/bin/env python3

import argparse

import subprocess

import rospkg
            
from codesign_pyutils.load_utils import LoadSols
from codesign_pyutils.misc_definitions import get_design_map

import matplotlib.pyplot as plt

import numpy as np

from codesign_pyutils.miscell_utils import extract_q_design, compute_man_measure, scatter3Dcodesign, select_best_sols

from codesign_pyutils.miscell_utils import Clusterer

import functools

def man_meas2_opt_cost(x, n_int):

    return x**2 * n_int

def opt_cost2_man_meas(x, n_int):

    return np.sqrt(x/n_int)

def main(args):

    # useful paths
    rospackage = rospkg.RosPack() # Only for taking the path to the leg package

    urdfs_path = rospackage.get_path("repair_urdf") + "/urdf"
    urdf_name = "repair_full"
    urdf_full_path = urdfs_path + "/" + urdf_name + ".urdf"
    xacro_full_path = urdfs_path + "/" + urdf_name + ".urdf.xacro"

    codesign_path = rospackage.get_path("repair_codesign")

    results_path = codesign_path + "/test_results/" + args.res_dirname + "/first_level"

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

    sol_loader = LoadSols(results_path)
    
    is_classical_man = bool(sol_loader.task_info_data["use_classical_man"])

    if is_classical_man:

        raise Exception("You still have to adapt this script to the classical man measure!!!!!!!!!!")

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
    fig_hist,ax_hist = plt.subplots()
    # =fig_hist.add_subplot(1, 1, 1)
    hist_plot = ax_hist.hist(man_measure, bins = int(n_opt_sol/20.0))
    # ax_hist.legend(loc="upper left")
    ax_hist.set_xlabel(r"perf. index[rad/s]")
    ax_hist.set_ylabel(r"N samples")
    # ax_hist.secondary_xaxis('top', functions=(functools.partial(man_meas2_opt_cost, n_int=n_int),
    #                                         functools.partial(opt_cost2_man_meas, n_int=n_int)))
    ax_hist.set_title(r"Performance index histogram", fontdict=None, loc='center')
    ax_hist.grid()
    # ax_hist.set_aspect('equal', 'box')
    # ax_hist.set_facecolor("#d3d3d3")
    # plt.show()
    
    green_diamond = dict(markerfacecolor='g', marker='D')
    fig_box, ax_box = plt.subplots(1)
    ax_box.boxplot(man_measure, flierprops = green_diamond, vert=True, 
                    # whis = (0, 100),
                    autorange = True)
    # leg = ax_sigma.legend(loc="upper left")
    # leg.set_draggable()
    ax_box.set_xlabel("cluster index")
    ax_box.set_ylabel("$\eta$\,[rad/s]")
    ax_box.set_title(r"Second level boxplot", fontdict=None, loc='center')
    ax_box.grid()

    # 3D scatterplots
    # opt_q_design_selections, opt_costs_sorted = select_best_sols(100, opt_costs, opt_q_design)
    # scatter3Dcodesign(opt_costs, opt_costs_sorted, opt_q_design_selections,  n_int)

    clusterer = Clusterer(opt_q_design.T, opt_costs, n_int, n_clusters = 40, 
                        algo_name = "minikmeans")
    
    clusterer.create_cluster_plot(show_clusters_sep = True, 
                                    show_cluster_costs = True, 
                                    plt_red_factor = 5)
    clusterer.show_plots()
    

if __name__ == '__main__':

    # adding script arguments
    parser = argparse.ArgumentParser(
        description='just a simple test file for RePAIR co-design')
    parser.add_argument('--res_dirname', '-d', type=str,\
                        help = 'directory name from where results are to be loaded', default = "load_dir")

    args = parser.parse_args()

    main(args)