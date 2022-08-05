
from turtle import right
import numpy as np

import os

from codesign_pyutils.misc_definitions import get_design_map

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

import time 

import matplotlib.pyplot as plt
from matplotlib import colors

from pylab import cm

from collections import Counter

def str2bool(v):
  #susendberg's function
  return v.lower() in ("yes", "true", "t", "1")

def wait_for_confirmation(do_something = "proceed",\
                          or_do_something_else = "stop",\
                          on_confirmation = "Confirmation received!",\
                          on_denial = "Stop received!"):

  usr_input = input("\n \n Press Enter to " + do_something + \
                    " or type \"N/n\" to " + or_do_something_else + ". \n -> ")

  if usr_input == "":

      print("\n" + on_confirmation + "\n")
      go_on = True
  
  else:

      if (usr_input == "no" or usr_input == "No" or usr_input == "N" or usr_input == "n"):
        
        print("\n" +  on_denial + "\n")
        go_on = False
        
      else: # user typed something else

        print("\n Ops.. you did not type any of the options! \n")

        go_on = wait_for_confirmation()

  return go_on

def check_str_list(comp_list = ["x", "y", "z"], input = []):
  
  presence_array = [False] * len(comp_list)

  for i in range(len(comp_list)):

    for j in range(len(input)):

      if (input[j] == comp_list[i] or input[j] == comp_list[i].upper()) and presence_array[i] != True:

        presence_array[i] = True

  return np.where(presence_array)[0]
  
def rot_error_axis_sel_not_supp(axis_selector, rot_error_approach):

  if len(axis_selector) != 3:
  
    raise Exception("\nSelecting the constrained axis when using \"" + rot_error_approach + "\" orientation error is not supported yet.\n")

def get_min_cost_index(costs, solve_failed_array):

    # to not solved probelms negative numbers are assignes

    best_index = - 1
    n_glb_tests = len(costs)

    if not solve_failed_array[0]:

        best_index = 0

    for i in range(1, n_glb_tests):

        if (not solve_failed_array[i]) and (best_index != -1): # if solution is valid and there exist a previous valid solution

          if (costs[i] - costs[best_index]) < 0:
            
            best_index = i
        
        if (not solve_failed_array[i]) and (best_index == -1): # no previous valid sol exists, but current solution is valid

          best_index = i        

    if best_index < 0:

        best_index = np.argmin(costs)

    return best_index

def get_full_abs_paths_sol(init_opt_results_path, init_failed_results_path):

  file_list_opt = os.listdir(init_opt_results_path)
  file_list_failed = os.listdir(init_failed_results_path)
  is_file_opt = [True] * len(file_list_opt) + [False] * len(file_list_failed)
  full_file_list = file_list_opt + file_list_failed
  full_file_paths = [""] * len(full_file_list)

  for i in range(len(full_file_list)):

    if is_file_opt[i]:

        full_file_paths[i] = init_opt_results_path + "/" + full_file_list[i]
    
    else:

        full_file_paths[i] = init_failed_results_path + "/" + full_file_list[i]

  return full_file_paths

def extract_q_design(input_data: list):

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

def compute_man_measure(opt_costs: list, n_int: int):

    man_measure = np.zeros((len(opt_costs), 1)).flatten()

    for i in range(len(opt_costs)): 

        man_measure[i] = np.sqrt(opt_costs[i] / n_int) # --> discretized root mean squared joint velocities over the opt interval 

    return man_measure

def scatter3Dcodesign(perc: float, opt_costs: list, opt_q_design: np.ndarray, n_int_prb: int, 
                      markersize = 20, use_abs_colormap_scale = True):

    perc = abs(float(perc))
    if perc > 100.0:
        perc = 100.0

    n_opt_sol = len(opt_costs)
    n_selection = round(n_opt_sol * perc/100)

    sorted_costs_indeces = np.argsort(np.array(opt_costs)) # indeces from best cost(lowest) to worst (biggest)
    selection_indeces = sorted_costs_indeces[:n_selection] # first n_selection indeces

    opt_q_design_selections = opt_q_design[:, selection_indeces]
    opt_costs_sorted = [opt_costs[i] for i in selection_indeces] 

    man_measure_original = compute_man_measure(opt_costs, n_int_prb)

    vmin_colorbar = None
    vmax_colorbar = None
    if use_abs_colormap_scale:
      vmin_colorbar = min(man_measure_original)
      vmax_colorbar = max(man_measure_original)

    man_measure_sorted = compute_man_measure(opt_costs_sorted, n_int_prb) # scaling opt costs to make them more interpretable

    fig = plt.figure()
    ax = plt.axes(projection ="3d")
    ax.grid(b = True, color ='grey',
        linestyle ='-.', linewidth = 0.3,
        alpha = 0.2)
    my_cmap = plt.get_cmap('jet_r')

    sctt = ax.scatter3D(opt_q_design_selections[0, :],\
                        opt_q_design_selections[1, :],\
                        opt_q_design_selections[2, :],\
                        alpha = 0.8,
                        c = man_measure_sorted.flatten(),
                        cmap = my_cmap,
                        marker ='o', 
                        s = markersize, 
                        vmin = vmin_colorbar, vmax = vmax_colorbar)
    plt.title("Co-design variables scatter plot - selection of " + str(int(n_selection/n_opt_sol * 100.0)) + "% of the best solutions")
    ax.set_xlabel('mount. height', fontweight ='bold')
    ax.set_ylabel('should. width', fontweight ='bold')
    ax.set_zlabel('mount. roll angle', fontweight ='bold')
    fig.colorbar(sctt, ax = ax, shrink = 0.5, aspect = 20, label='performance index')

    return True

class Clusterer():

  def __init__(self, X, n_clusters = 5):

    self.base_options = {
    "n_neighbors": 3,
    "n_clusters": n_clusters,
    "min_samples": 20,
    "metric": "euclidean", 
    "max_neigh_sample_dist": 0.01,
    }

    self.X = X

    self.connectivity = kneighbors_graph(
        self.X, n_neighbors=self.base_options["n_neighbors"], include_self=False
    )

    self.connectivity_mat = 0.5 * (self.connectivity + self.connectivity.T)

    self.algo_dict = {}

    self.algo_dict["kmeans"] = cluster.KMeans(n_clusters=self.base_options["n_clusters"])

    self.algo_dict["minikmeans"] = cluster.MiniBatchKMeans(n_clusters=self.base_options["n_clusters"])

    self.algo_dict["bi_kmeans"] = cluster.BisectingKMeans(n_clusters=self.base_options["n_clusters"])

    self.algo_dict["agg_cl_ward"] = cluster.AgglomerativeClustering(n_clusters=self.base_options["n_clusters"],
                                                  linkage="ward",
                                                  connectivity=None, 
                                                  distance_threshold = 0.4)

    # self.algo_dict["birk"] = cluster.Birch(n_clusters=self.base_options["n_clusters"])

    self.algo_dict["optics"] = cluster.OPTICS(min_samples = self.base_options["min_samples"], 
                                              metric = self.base_options["metric"], 
                                              eps = self.base_options["max_neigh_sample_dist"])

    self.algo_dict["dbscan"] = cluster.DBSCAN(eps = self.base_options["max_neigh_sample_dist"], 
                                              min_samples = self.base_options["min_samples"], 
                                              metric = self.base_options["metric"], 
                                              )

  def compute_clust(self, method_name = "kmeans"):
    
    algorithm = self.algo_dict[method_name]

    t0 = time.time()
    model = algorithm.fit(self.X)
    t1 = time.time()

    if hasattr(algorithm, "labels_"):

        y_pred = algorithm.labels_.astype(int)

    else:

        y_pred = algorithm.predict(self.X)

    n_clusters = len(Counter(y_pred).keys())

    print("Number of clusters set/found: " +  str(n_clusters))
    
    clust_indeces, cluster_size_vector = self.get_cluster_sizes(y_pred)
    print("Cluster size vector: " +  str(cluster_size_vector))
    print("Cluster indeces" + str(clust_indeces)) 
    print(np.sum(cluster_size_vector))

    return y_pred

  def get_rbg(self, y):

    n_clusters = len(Counter(y).keys())    
    cmap = cm.get_cmap('seismic', n_clusters)
    rbg = []
    for i in range(cmap.N):
        rbg.append(colors.rgb2hex(cmap(i)))

    return np.array(rbg)

  def get_cluster_sizes(self, y):

    y_un = np.unique(y) # vector of unique cluster IDs

    cl_size_vector = []

    for i in range(len(y_un)):

      cl_size_vector.append(len(np.where(y == y_un[i])[0]))

    return y_un, np.array(cl_size_vector) 
  
  def create_cluster_plot(self, method_name = "kmeans", show_clusters_sep = False):
    
    y = self.compute_clust(method_name)

    rgb_colors = self.get_rbg(y)

    plt.figure()

    ax = plt.axes(projection ="3d")
    ax.grid(b = True, color ='grey',
        linestyle ='-.', linewidth = 0.3,
        alpha = 0.2)

    ax.scatter3D(self.X[:, 0],\
                  self.X[:, 1],\
                  self.X[:, 2],\
                  alpha = 0.8,
                  c = rgb_colors[y],
                  marker ='o', 
                  s = 30)

    plt.title("Co-design variables scatter plot - clustering with " + method_name)
    ax.set_xlabel('mount. height', fontweight ='bold')
    ax.set_ylabel('should. width', fontweight ='bold')
    ax.set_zlabel('mount. roll angle', fontweight ='bold')
    
    if show_clusters_sep:

      y_un = np.unique(y)

      for i in range(len(y_un)):
        
        plt.figure()
      
        cluster_selector = np.where(y == y_un[i])[0]

        ax = plt.axes(projection ="3d")
        ax.set_xlim3d(np.min(self.X[:, 0]), np.max(self.X[:, 0]))
        ax.set_ylim3d(np.min(self.X[:, 1]), np.max(self.X[:, 1]))
        ax.set_zlim3d(np.min(self.X[:, 2]), np.max(self.X[:, 2]))

        ax.grid(b = True, color ='grey',  
            linestyle ='-.', linewidth = 0.3,
            alpha = 0.2)

        ax.scatter3D(self.X[cluster_selector, 0],\
                      self.X[cluster_selector, 1],\
                      self.X[cluster_selector, 2],\
                      alpha = 0.8,
                      c = rgb_colors[[y_un[i]] * len(cluster_selector)],
                      marker ='o', 
                      s = 30)

        plt.title("Co-design variables scatter plot - clustering with " + method_name + "/" + str(y_un[i]))
        ax.set_xlabel('mount. height', fontweight ='bold')
        ax.set_ylabel('should. width', fontweight ='bold')
        ax.set_zlabel('mount. roll angle', fontweight ='bold')

  def show_plots(self):

    plt.show()

  def get_algo_names(self):

    return list(self.algo_dict.keys())

        