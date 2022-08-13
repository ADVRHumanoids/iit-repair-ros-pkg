
from re import X
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
        design_var_map["should_w_l"],\
        design_var_map["should_roll_l"],\
        design_var_map["wrist_off_l"]]

    # design_indeces = [design_var_map["mount_h"],\
    #     design_var_map["should_w_l"],\
    #     design_var_map["should_roll_l"]]

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

def gen_y_sampling(n_y_samples, y_sampl_ub):

    y_sampl_lb = - y_sampl_ub
    if n_y_samples == 1:
        dy = 0.0
    else:
        dy = (y_sampl_ub - y_sampl_lb) / (n_y_samples - 1)

    y_sampling = np.array( [0.0] * n_y_samples)
    for i in range(n_y_samples):
        
        y_sampling[i] = y_sampl_lb + dy * i

    return y_sampling

def compute_solution_divs(n_multistrt: int, n_prcss: int):
    
    n_sol_tries = n_multistrt
    n_p = n_prcss

    n_divs = int(np.floor(n_sol_tries / n_p)) 

    n_remaining_sols = n_sol_tries - n_divs * n_p

    opt_divs = []

    if n_multistrt <= n_prcss: #assign a multistart to each process

      for i in range(n_multistrt):

        opt_divs.append(list(range(n_divs * i, n_divs * i + n_divs)))
        
    else: # distribute them appropiately

      for i in range(n_p):

          opt_divs.append(list(range(n_divs * i, n_divs * i + n_divs)))

    if n_remaining_sols != 0:
        
        for i in range(n_remaining_sols):

            opt_divs[i].append(n_divs * (n_p - 1) + n_divs + i)

        # opt_divs = [[]] * (n_p + 1)

        # for i in range(n_p + 1):
            
        #     if i == n_p:

        #         opt_divs[i] = list(range(n_divs * i, n_divs * i + n_remaining_sols))

        #     else:

        #         opt_divs[i] = list(range(n_divs * i, n_divs * i + n_divs))

    return opt_divs

def select_best_sols(perc: float, opt_costs: list, opt_q_design: np.ndarray):

  perc = abs(float(perc))
  if perc > 100.0:
      perc = 100.0

  n_opt_sol = len(opt_costs)
  n_selection = round(n_opt_sol * perc/100)

  sorted_costs_indeces = np.argsort(np.array(opt_costs)) # indeces from best cost(lowest) to worst (biggest)
  selection_indeces = sorted_costs_indeces[:n_selection] # first n_selection indeces

  opt_q_design_selections = opt_q_design[:, selection_indeces]
  opt_costs_sorted = [opt_costs[i] for i in selection_indeces]

  return opt_q_design_selections, opt_costs_sorted

def scatter3Dcodesign(opt_costs: list,
                      opt_costs_sorted: list, opt_q_design_selections: np.ndarray,
                      n_int_prb: int, 
                      markersize = 20, use_abs_colormap_scale = True):

    n_selection = len(opt_costs_sorted)
    n_opt_sol = len(opt_costs)

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

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()
        
class Clusterer():

  def __init__(self, X: np.ndarray, opt_costs: list,
              n_int: int,
              n_clusters=5, mark_dim=30,
              algo_name="minikmeans"):

    self.base_options = {
    "n_neighbors": 3,
    "n_clusters": n_clusters,
    "min_samples": 32,
    "metric": "euclidean", 
    "max_neigh_sample_dist": 0.01,
    "distance_threshold_ward": 0.4,
    "random_seed": 1, # crucial to make clustering deterministic
    "n_int": 50,
    }

    self.X = X
    self.opt_costs = opt_costs
    self.n_int = n_int
    self.man_measure = compute_man_measure(self.opt_costs, self.n_int)

    self.vmin_colorbar = min(self.man_measure)
    self.vmax_colorbar = max(self.man_measure)

    self.mark_dim = mark_dim

    self.connectivity = kneighbors_graph(
        self.X, n_neighbors=self.base_options["n_neighbors"], include_self=False
    )

    self.connectivity_mat = 0.5 * (self.connectivity + self.connectivity.T)

    self.algo_dict = {}

    self.algo_dict["kmeans"] = cluster.KMeans(n_clusters=self.base_options["n_clusters"], 
                                              n_init=self.base_options["n_int"], 
                                              random_state=self.base_options["random_seed"])

    self.algo_dict["minikmeans"] = cluster.MiniBatchKMeans(n_clusters=self.base_options["n_clusters"],
                                                          n_init=self.base_options["n_int"], 
                                                          random_state=self.base_options["random_seed"])

    self.algo_dict["bi_kmeans"] = cluster.BisectingKMeans(n_clusters=self.base_options["n_clusters"],
                                                          n_init=self.base_options["n_int"], 
                                                          random_state=self.base_options["random_seed"])

    if self.base_options["n_clusters"] is None:

      self.algo_dict["agg_cl_ward"] = cluster.AgglomerativeClustering(n_clusters=self.base_options["n_clusters"],
                                                    linkage="ward",
                                                    connectivity=None, 
                                                    distance_threshold = self.base_options["distance_threshold_ward"])
    else:

      self.algo_dict["agg_cl_ward"] = cluster.AgglomerativeClustering(n_clusters=self.base_options["n_clusters"],
                                                    linkage="ward",
                                                    connectivity=None)

    # self.algo_dict["birk"] = cluster.Birch(n_clusters=self.base_options["n_clusters"])

    self.algo_dict["optics"] = cluster.OPTICS(min_samples = self.base_options["min_samples"], 
                                              metric = self.base_options["metric"], 
                                              eps = self.base_options["max_neigh_sample_dist"])

    self.algo_dict["dbscan"] = cluster.DBSCAN(eps = self.base_options["max_neigh_sample_dist"], 
                                              min_samples = self.base_options["min_samples"], 
                                              metric = self.base_options["metric"], 
                                              )

  def get_clust_ids(self):

    return self.clust_ids
  
  def get_cl_size_vect(self):

    return self.cl_size_vect
 
  def get_cluster_selector(self, cl_index: int):

    cluster_selector = np.where(self.data_clust_array == cl_index)[0]

    return cluster_selector

  def get_clust_costs(self, cl_index: int):

    cluster_selector = self.get_cluster_selector(cl_index)

    cl_costs = [self.opt_costs[i] for i in cluster_selector]

    return np.array(cl_costs)

  def get_cluster_data(self, cl_index: int):

    cluster_selector = self.get_cluster_selector(cl_index)

    X_sel = self.X[cluster_selector, :]

    return X_sel

  def compute_first_level_candidates(self):

    opt_index_abs = [-1] * self.n_clust

    for i in range(self.n_clust):

      cl_costs_i = self.get_clust_costs(i)
      cl_selector_i = self.get_cluster_selector(i)
      self.man_measure

      opt_index = np.argwhere(cl_costs_i == np.min(cl_costs_i))[0][0]
      opt_index_abs[i] = cl_selector_i[opt_index]

    return opt_index_abs

  def get_fist_lev_candidate_man_measure(self):

    opt_index_abs = self.compute_first_level_candidates()

    return self.man_measure[opt_index_abs]

  def get_fist_lev_candidate_opt_cost(self):

    opt_index_abs = self.compute_first_level_candidates()

    return np.array(self.opt_costs)[opt_index_abs]

  def get_algo_names(self):

    return list(self.algo_dict.keys())
  
  def get_n_clust(self):

    return self.n_clust

  def clusterize(self, algo_name="minikmeans"):

    self.data_clust_array = self.compute_clust(algo_name)

    self.clust_ids,  self.cl_size_vect = self.get_cluster_sizes(self.data_clust_array)

    self.n_clust = len(self.clust_ids)

    self.compute_first_level_candidates()

  def compute_clust(self, method_name="minikmeans"):
    
    algorithm = self.algo_dict[method_name]

    algorithm.fit(self.X)

    if hasattr(algorithm, "labels_"):

        y_pred = algorithm.labels_.astype(int)

    else:

        y_pred = algorithm.predict(self.X)

    n_clusters = len(Counter(y_pred).keys())

    print("Number of clusters set/found: " +  str(n_clusters))
    
    clust_indeces, cluster_size_vector = self.get_cluster_sizes(y_pred)
    print("Cluster size vector: " +  str(cluster_size_vector))

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
  
  def create_cluster_plot(self, method_name = "minikmeans",
                          show_clusters_sep = False, 
                          show_background_pnts = True, 
                          show_cluster_costs = False, 
                          plt_red_factor = 1,
                          show_leg = True):
    
    y_un = np.unique(self.data_clust_array)
    clust_plt_ind = list(range(0, int(len(y_un)/plt_red_factor)))

    rgb_colors = self.get_rbg(self.data_clust_array)

    plt.figure()

    ax = plt.axes(projection ="3d")
    ax.grid(b = True, color ='grey',
        linestyle ='-.', linewidth = 0.3,
        alpha = 0.2)

    for i in clust_plt_ind:

      cluster_selector = np.where(self.data_clust_array == y_un[i])[0]

      ax.scatter3D(self.X[cluster_selector, 0],\
                    self.X[cluster_selector, 1],\
                    self.X[cluster_selector, 2],\
                    alpha = 0.8,
                    c = rgb_colors[self.data_clust_array[i]],
                    marker ='o', 
                    s = self.mark_dim, 
                    label="cluster n." + str(y_un[i]))

    if show_background_pnts:

      ax.scatter3D(self.X[:, 0],\
                    self.X[:, 1],\
                    self.X[:, 2],\
                    alpha = 0.05,
                    c = ["#D3D3D3"] * len(self.opt_costs),
                    marker ='o', 
                    s = self.mark_dim)

    if show_leg:
      leg = ax.legend(loc="upper left")
      leg.set_draggable(True)

    plt.title("Co-design variables scatter plot - clustering with " + method_name)
    ax.set_xlabel('mount. height', fontweight ='bold')
    ax.set_ylabel('should. width', fontweight ='bold')
    ax.set_zlabel('mount. roll angle', fontweight ='bold')
    
    if show_clusters_sep:

      for i in clust_plt_ind:
        
        fig = plt.figure()
      
        cluster_selector = np.where(self.data_clust_array == y_un[i])[0]
        rest_of_points_selector = np.where(self.data_clust_array != y_un[i])[0]

        ax = plt.axes(projection ="3d")
        ax.set_xlim3d(np.min(self.X[:, 0]), np.max(self.X[:, 0]))
        ax.set_ylim3d(np.min(self.X[:, 1]), np.max(self.X[:, 1]))
        ax.set_zlim3d(np.min(self.X[:, 2]), np.max(self.X[:, 2]))

        ax.grid(b = True, color ='grey',  
            linestyle ='-.', linewidth = 0.3,
            alpha = 0.2)

        if show_background_pnts:

          ax.scatter3D(self.X[rest_of_points_selector, 0],\
                        self.X[rest_of_points_selector, 1],\
                        self.X[rest_of_points_selector, 2],\
                        alpha = 0.05,
                        c = ["#D3D3D3"] * len(rest_of_points_selector),
                        marker ='o', 
                        s = self.mark_dim)

        colrs = rgb_colors[[y_un[i]] * len(cluster_selector)]

        if not show_cluster_costs: 

          ax.scatter3D(self.X[cluster_selector, 0],\
                      self.X[cluster_selector, 1],\
                      self.X[cluster_selector, 2],\
                      alpha = 1,
                      c = colrs,
                      marker ='o', 
                      s = self.mark_dim, 
                      label="cluster n." + str(y_un[i]))
        
        else:
          
          man_measure_clust = compute_man_measure([self.opt_costs[i] for i in cluster_selector],
                                             self.n_int)

          my_cmap = plt.get_cmap('jet_r')

          sctt = ax.scatter3D(self.X[cluster_selector, 0],\
                              self.X[cluster_selector, 1],\
                              self.X[cluster_selector, 2],\
                              alpha = 0.8,
                              c = man_measure_clust.flatten(),
                              cmap = my_cmap,
                              marker ='o', 
                              s = 30, 
                              vmin = self.vmin_colorbar, vmax = self.vmax_colorbar, 
                              label="cluster n." + str(y_un[i]))

          fig.colorbar(sctt, ax = ax, shrink = 0.5, aspect = 20, label='performance index')

        if show_leg:
          leg = ax.legend(loc="upper left")
          leg.set_draggable(True)
        
        plt.title("Clustering method: " + method_name + 
        "\n Cluster size:" + str(len(cluster_selector)))
        ax.set_xlabel('mount. height', fontweight ='bold')
        ax.set_ylabel('should. width', fontweight ='bold')
        ax.set_zlabel('mount. roll angle', fontweight ='bold')

  def show_plots(self):

    plt.show()



        