#!/usr/bin/env python3
import argparse

import rospkg
        
from codesign_pyutils.post_proc_utils import PostProc2ndLev

import numpy as np

import matplotlib.pyplot as plt

def main(args):

    green_diamond = dict(markerfacecolor='g', marker='D')

    # useful paths
    rospackage = rospkg.RosPack() # Only for taking the path to the leg package
    codesign_path = rospackage.get_path("repair_codesign")

    results_path = codesign_path + "/test_results/" + args.res_dirname

    dump_dirname = "2nd_lev_postproc"

    post_proc = PostProc2ndLev(results_path, 
                    dump_dirname = dump_dirname) # loads 2nd level solutions and dumps general and postproc. info to file

    opt_costs = post_proc.second_lev_opt_costs
    man_meass = post_proc.second_lev_man_measure

    plt_red_factor = 2
    clust_plt_ind = list(range(0, int(post_proc.n_clust/plt_red_factor)))
    fig_hist = [None] * len(clust_plt_ind)
    ax_hist = [None] * len(clust_plt_ind)
    fig_scatt = [None] * len(clust_plt_ind)
    ax_scatt = [None] * len(clust_plt_ind)
    i = 0
    for cl_idx in clust_plt_ind:

        fig_hist[i], ax_hist[i] = plt.subplots()
        ax_hist[i].hist(np.array(man_meass[cl_idx]), bins = int(len(man_meass[cl_idx])/3))
        ax_hist[i].set_xlabel(r"man. cost[rad/s]")
        ax_hist[i].set_ylabel(r"N samples")
        ax_hist[i].set_title(r"Performance index histogram clust n." + str(cl_idx), fontdict=None, loc='center')
        ax_hist[i].grid()

        fig_scatt[i], ax_scatt[i] = plt.subplots()
        ax_scatt[i].scatter(np.array(list(range(len(man_meass[cl_idx])))),\
            np.array(man_meass[cl_idx]), label=r"refinement points", marker="o", s=50 )
        ax_scatt[i].scatter(0, post_proc.first_lev_man_measure[cl_idx], label=r"perf. index initial guess", marker="o", s=50 , c = "#ffa500")
        leg = ax_scatt[i].legend(loc="upper left")
        leg.set_draggable(True)
        ax_scatt[i].set_xlabel("mutlistart index")
        ax_scatt[i].set_ylabel("perf. index[rad/s]")
        ax_scatt[i].set_title(r"Clust n." + str(cl_idx), fontdict=None, loc='center')
        ax_scatt[i].grid()

        i = i + 1

    fig_sigma, ax_sigma = plt.subplots(2)
    ax_sigma[0].scatter(np.array(list(range(post_proc.n_clust))),\
        np.array(post_proc.rmse_man_meas), label=r"", marker="o", s=50 )
    # leg = ax_sigma.legend(loc="upper left")
    # leg.set_draggable()
    ax_sigma[0].set_xlabel("cluster index")
    ax_sigma[0].set_ylabel("$\sigma_{cl}$\,[rad/s]")
    ax_sigma[0].set_title(r"RMSE", fontdict=None, loc='center')
    ax_sigma[0].grid()
    ax_sigma[1].scatter(np.array(list(range(post_proc.n_clust))),\
        np.array(post_proc.second_lev_true_man), label=r"", marker="o", s=50 )
    # leg = ax_sigma.legend(loc="upper left")
    # leg.set_draggable()
    ax_sigma[1].set_xlabel("cluster index")
    ax_sigma[1].set_ylabel("$\sigma_{cl}$\,[rad/s]")
    ax_sigma[1].set_title(r"RMSE", fontdict=None, loc='center')
    ax_sigma[1].grid()

    fig_box, ax_box = plt.subplots(1)
    ax_box.boxplot(man_meass, flierprops = green_diamond, vert=True, 
                    # whis = (0, 100),
                    autorange = True)
    # leg = ax_sigma.legend(loc="upper left")
    # leg.set_draggable()
    ax_box.set_xlabel("cluster index")
    ax_box.set_ylabel("$\eta$\,[rad/s]")
    ax_box.set_title(r"Second level boxplot", fontdict=None, loc='center')
    ax_box.grid()


    plt.show()

if __name__ == '__main__':

    # adding script arguments
    parser = argparse.ArgumentParser(
        description='just a simple test file for RePAIR co-design')
    parser.add_argument('--res_dirname', '-d', type=str,\
                        help = 'directory name from where results are to be loaded', default = "load_dir")

    args = parser.parse_args()

    main(args)