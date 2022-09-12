#!/usr/bin/env python3

## Script for converting mat files containing trajectories to CSV ##

from post_process_utils import LogLoader, LogPlotter

mat_path = "/home/andreap/hhcm_ws/src/iit-repair-ros-pkg/repair_codesign/test_results/test_results_07-09-2022-16_52_10/l2_postproc/"
mat_name = "TrajReplayerRt__0_2022_09_12__12_11_26"

data_loader = LogLoader(mat_path + mat_name + ".mat")
data = data_loader.data

plotter = LogPlotter(data)
plotter.make_plots()
plotter.show_plots()


