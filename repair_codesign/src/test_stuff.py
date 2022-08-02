#!/usr/bin/env python3

# from codesign_pyutils.load_utils import LoadSols

# import rospkg

# useful paths
# rospackage = rospkg.RosPack() # Only for taking the path to the leg package

# urdfs_path = rospackage.get_path("repair_urdf") + "/urdf"
# urdf_name = "repair_full"
# urdf_full_path = urdfs_path + "/" + urdf_name + ".urdf"
# xacro_full_path = urdfs_path + "/" + urdf_name + ".urdf.xacro"

# codesign_path = rospackage.get_path("repair_codesign")

# results_path = codesign_path + "/test_results"

# replay_folder_name = "replay_directory"
# replay_opt_results_path = results_path + "/" + replay_folder_name + "/opt" 
# replay_failed_results_path = results_path + "/initializations/" + replay_folder_name + "/failed" 
# sol_info_path = results_path + "/" + replay_folder_name

# sol_loader = LoadSols(sol_info_path)


# print(sol_loader.fail_data)

# from codesign_pyutils.ros_utils import MarkerGen


# lft_trgt_marker_topic = "repair/lft_trgt"
# rviz_marker_gen = MarkerGen(node_name = "marker_gen_cart_task")
# rviz_marker_gen.add_marker("working_surface_link", [0.0, 0.3, 0.4], lft_trgt_marker_topic,\
#                         "left_trgt", 0.3) 
# rviz_marker_gen.spin()

# from codesign_pyutils.miscell_utils import scatter3Dcodesign, compute_man_measure
# import numpy as np
# import matplotlib.pyplot as plt
# opt_costs = [0.3, 0.02, 0.3, 0.35, 0.493, 0.1]
# q_design_opt = np.array([[0.0, 0.5, 1, 1.5, 2.0, 0.0], [0.0, 0.5, 1, 1.5, 2.0, 0.0], [0.0, 0.5, 1, 1.5, 2.0, 0.0]])

# man_measure = compute_man_measure(opt_costs, 5)
# fig = plt.figure()
# ax = plt.axes(projection ="3d")
# ax.grid(b = True, color ='grey',
#     linestyle ='-.', linewidth = 0.3,
#     alpha = 0.2)
# my_cmap = plt.get_cmap('jet_r')

# sctt = ax.scatter3D(q_design_opt[0, :],\
#                     q_design_opt[1, :],\
#                     q_design_opt[2, :],\
#                     alpha = 0.8,
#                     c = man_measure.flatten(),
#                     cmap = my_cmap,
#                     marker ='o', 
#                     s = 100)
# plt.title("Co-design variables scatter plot")
# ax.set_xlabel('mount. height', fontweight ='bold')
# ax.set_ylabel('should. width', fontweight ='bold')
# ax.set_zlabel('mount. roll angle', fontweight ='bold')
# fig.colorbar(sctt, ax = ax, shrink = 0.5, aspect = 20, label='performance index')

# scatter3Dcodesign(100, opt_costs, q_design_opt, 5, markersize = 100)
plt.show()