#!/usr/bin/env python3

from codesign_pyutils.load_utils import LoadSols

import rospkg

# useful paths
rospackage = rospkg.RosPack() # Only for taking the path to the leg package

urdfs_path = rospackage.get_path("repair_urdf") + "/urdf"
urdf_name = "repair_full"
urdf_full_path = urdfs_path + "/" + urdf_name + ".urdf"
xacro_full_path = urdfs_path + "/" + urdf_name + ".urdf.xacro"

codesign_path = rospackage.get_path("repair_codesign")

results_path = codesign_path + "/test_results"

replay_folder_name = "replay_directory"
replay_opt_results_path = results_path + "/" + replay_folder_name + "/opt" 
replay_failed_results_path = results_path + "/initializations/" + replay_folder_name + "/failed" 
sol_info_path = results_path + "/" + replay_folder_name

sol_loader = LoadSols(sol_info_path)


print(sol_loader.fail_data)