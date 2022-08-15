import matplotlib.pyplot as plt

from codesign_pyutils.math_utils import compute_man_index

def scatter3Dcodesign(opt_costs: list,
                      opt_costs_sorted: list, opt_q_design_selections: np.ndarray,
                      n_int_prb: int, 
                      markersize = 20, use_abs_colormap_scale = True):

    n_selection = len(opt_costs_sorted)
    n_opt_sol = len(opt_costs)

    man_measure_original = compute_man_index(opt_costs, n_int_prb)

    vmin_colorbar = None
    vmax_colorbar = None
    if use_abs_colormap_scale:
      vmin_colorbar = min(man_measure_original)
      vmax_colorbar = max(man_measure_original)

    man_measure_sorted = compute_man_index(opt_costs_sorted, n_int_prb) # scaling opt costs to make them more interpretable

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

