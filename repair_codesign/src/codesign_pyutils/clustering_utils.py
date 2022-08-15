
from matplotlib import colors

from pylab import cm

from collections import Counter

from sklearn import cluster
from sklearn.neighbors import kneighbors_graph

from codesign_pyutils.math_utils import compute_man_index

import numpy as np

import matplotlib.pyplot as plt

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
    self.man_measure = compute_man_index(self.opt_costs, self.n_int)

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
    self.clusterize()

    self.separate_clust_man_measure()

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

  def separate_clust_man_measure(self):

    y_un = np.unique(self.data_clust_array)

    self.clusts_man_meas = [-1.0] * self.n_clust
    self.clusts_opt_costs = [-1.0] * self.n_clust

    for cl in range(self.n_clust):

      cluster_selector = np.where(self.data_clust_array == y_un[cl])[0]
      self.clusts_opt_costs[cl] = [self.opt_costs[i] for i in cluster_selector]
      self.clusts_man_meas[cl] = compute_man_index(self.clusts_opt_costs[cl],
                                                      self.n_int)
                                                      
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
    
    _1, cluster_size_vector = self.get_cluster_sizes(y_pred)
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

    # if show_background_pnts:

    #   ax.scatter3D(self.X[:, 0],\
    #                 self.X[:, 1],\
    #                 self.X[:, 2],\
    #                 alpha = 0.05,
    #                 c = ["#D3D3D3"] * len(self.opt_costs),
    #                 marker ='o', 
    #                 s = self.mark_dim)

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

        colrs = rgb_colors[[y_un[i]] * len(self.clusts_man_meas[i])]

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

          my_cmap = plt.get_cmap('jet_r')

          sctt = ax.scatter3D(self.X[cluster_selector, 0],\
                              self.X[cluster_selector, 1],\
                              self.X[cluster_selector, 2],\
                              alpha = 0.8,
                              c = self.clusts_man_meas[i].flatten(),
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

    green_diamond = dict(markerfacecolor='g', marker='D')
    _, ax_box = plt.subplots(1)
    ax_box.boxplot(self.clusts_man_meas, flierprops = green_diamond, vert=True, 
                    # whis = (0, 100),
                    autorange = True)
    ax_box.set_xlabel("cluster index")
    ax_box.set_ylabel("$\eta$\,[rad/s]")
    ax_box.set_title(r"First level cluster boxplot", fontdict=None, loc='center')
    ax_box.grid()
      
  def show_plots(self):

    plt.show()
