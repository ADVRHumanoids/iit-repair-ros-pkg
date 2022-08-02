#!/usr/bin/env python3

import argparse

import subprocess

import rospkg

from codesign_pyutils.miscell_utils import str2bool
            
from codesign_pyutils.tasks import TaskGen
from codesign_pyutils.load_utils import LoadSols
from codesign_pyutils.misc_definitions import get_design_map

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

import numpy as np

from codesign_pyutils.miscell_utils import extract_q_design, compute_man_measure, scatter3Dcodesign

from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice

import time

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

    # 1D histogram (w.r.t. perfomance index) 
    # plt.figure()
    # plt.hist(man_measure, bins = 200)
    # plt.legend(loc="upper left")
    # plt.xlabel(r"rad/s")
    # plt.ylabel(r"N. sol")
    # plt.title(r"Cost histogram", fontdict=None, loc='center')
    # plt.grid()

    # 3D scatterplots
    scatter3Dcodesign(100, opt_costs, opt_q_design, n_int)
    # scatter3Dcodesign(80, opt_costs, opt_q_design, n_int)
    # scatter3Dcodesign(60, opt_costs, opt_q_design, n_int)
    # scatter3Dcodesign(40, opt_costs, opt_q_design, n_int)
    # scatter3Dcodesign(20, opt_costs, opt_q_design, n_int)
    # scatter3Dcodesign(10, opt_costs, opt_q_design, n_int)
    # scatter3Dcodesign(5, opt_costs, opt_q_design, n_int)
    # scatter3Dcodesign(1, opt_costs, opt_q_design, n_int)

    # clustering test
    
    options = {
    "quantile": 0.3,
    "eps": 0.3,
    "damping": 0.9,
    "preference": -200,
    "n_neighbors": 3,
    "n_clusters": 10,
    "min_samples": 7,
    "xi": 0.05,
    "min_cluster_size": 0.1,
    }

    X = opt_q_design.T # data needs to be arranged as [samples x n_features]
    # X = StandardScaler().fit_transform(X) # normalize dataset for easier parameter selection

    bandwidth = cluster.estimate_bandwidth(X, quantile=options["quantile"])

    connectivity = kneighbors_graph(
        X, n_neighbors=options["n_neighbors"], include_self=False
    )

    # make connectivity symmetric
    connectivity = 0.5 * (connectivity + connectivity.T)

    # ============
    # Create cluster objects
    # ============
    ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
    two_means = cluster.MiniBatchKMeans(n_clusters=options["n_clusters"])
    ward = cluster.AgglomerativeClustering(
        n_clusters=options["n_clusters"], linkage="ward", connectivity=connectivity
    )
    spectral = cluster.SpectralClustering(
        n_clusters=options["n_clusters"],
        eigen_solver="arpack",
        affinity="nearest_neighbors",
    )
    dbscan = cluster.DBSCAN(eps=options["eps"])
    optics = cluster.OPTICS(
        min_samples=options["min_samples"],
        xi=options["xi"],
        min_cluster_size=options["min_cluster_size"],
    )
    affinity_propagation = cluster.AffinityPropagation(
        damping=options["damping"], preference=options["preference"], random_state=0
    )
    average_linkage = cluster.AgglomerativeClustering(
        linkage="average",
        affinity="cityblock",
        n_clusters=options["n_clusters"],
        connectivity=connectivity,
    )
    birch = cluster.Birch(n_clusters=options["n_clusters"])
    gmm = mixture.GaussianMixture(
        n_components=options["n_clusters"], covariance_type="full"
    )

    clustering_algorithms = (
        ("MiniBatch\nKMeans", two_means),
        ("Affinity\nPropagation", affinity_propagation),
        ("MeanShift", ms),
        ("Spectral\nClustering", spectral),
        ("Ward", ward),
        ("Agglomerative\nClustering", average_linkage),
        ("DBSCAN", dbscan),
        ("OPTICS", optics),
        ("BIRCH", birch),
        ("Gaussian\nMixture", gmm),
    )

    t0 = time.time()
    spectral.fit(X)
    
    t1 = time.time()
    if hasattr(spectral, "labels_"):
        y_pred = spectral.labels_.astype(int)
    else:
        y_pred = spectral.predict(X)

    colors = np.array(
        list(
            islice(
                cycle(
                    [
                        "#377eb8",
                        "#ff7f00",
                        "#4daf4a",
                        "#f781bf",
                        "#a65628",
                        "#984ea3",
                        "#999999",
                        "#e41a1c",
                        "#dede00",
                    ]
                ),
                int(max(y_pred) + 1),
            )
        )
    )
    # add black color for outliers (if any)
    colors = np.append(colors, ["#000000"])

    fig = plt.figure()
    ax = plt.axes(projection ="3d")
    ax.grid(b = True, color ='grey',
        linestyle ='-.', linewidth = 0.3,
        alpha = 0.2)
    my_cmap = plt.get_cmap('jet_r')

    sctt = ax.scatter3D(X[:, 0],\
                        X[:, 1],\
                        X[:, 2],\
                        alpha = 0.8,
                        c = colors[y_pred],
                        cmap = my_cmap,
                        marker ='o', 
                        s = 30)

    plt.title("Co-design variables scatter plot - clustering ")
    ax.set_xlabel('mount. height', fontweight ='bold')
    ax.set_ylabel('should. width', fontweight ='bold')
    ax.set_zlabel('mount. roll angle', fontweight ='bold')
    fig.colorbar(sctt, ax = ax, shrink = 0.5, aspect = 20, label='performance index')

    plt.show()

if __name__ == '__main__':

    # adding script arguments
    parser = argparse.ArgumentParser(
        description='just a simple test file for RePAIR co-design')
    parser.add_argument('--resample_sol', '-rs', type=str2bool,\
                        help = 'whether to resample the obtained solution before replaying it', default = False)

    args = parser.parse_args()

    main(args)