#!/usr/bin/env python3

from codesign_pyutils.post_proc_utils import PostProcL1, PostProcL2

import numpy as np

# postprl1 = PostProcL1("/home/andreap/hhcm_ws/src/iit-repair-ros-pkg/repair_codesign/test_results/load_dir",
#                         cl_man_post_proc= False)

# postprl1.print_sol_run_info()
# postprl1.clusterize(40)
# postprl1.make_plots()
# postprl1.show_plots()

postprl2 = PostProcL2("/home/andreap/hhcm_ws/src/iit-repair-ros-pkg/repair_codesign/test_results/test_results_31-08-2022-13_10_21") 