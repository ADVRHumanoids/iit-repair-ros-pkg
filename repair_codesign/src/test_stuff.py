#!/usr/bin/env python3

from codesign_pyutils.post_proc_utils import PostProcS1, PostProcS3
from codesign_pyutils.misc_definitions import get_design_map, get_coll_joint_map

import numpy as np

# postprl1 = PostProcS1("/home/andreap/hhcm_ws/src/iit-repair-ros-pkg/repair_codesign/test_results/test_results_12-09-2022-17_44_36",
#                         cl_man_post_proc= False)

# postprl1.print_sol_run_info()
# postprl1.clusterize(10)
# postprl1.make_plots(bin_scale_factor=6, plt_red_factor = 1)
# postprl1.show_plots()


postprl2 = PostProcS3("/home/andreap/hhcm_ws/src/iit-repair-ros-pkg/repair_codesign/test_results/test_results_12-09-2022-17_44_36") 