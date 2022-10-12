#!/usr/bin/env python3

from codesign_pyutils.post_proc_utils import PostProcS1, PostProcS3
from codesign_pyutils.misc_definitions import get_design_map, get_coll_joint_map

import numpy as np

postprs1 = PostProcS1("/home/andreap/hhcm_ws/src/iit-repair-ros-pkg/repair_codesign/test_results/test_results_11-10-2022-14_19_55",
                        cl_man_post_proc= False)

postprs1.print_sol_run_info()
postprs1.clusterize(10)
postprs1.make_plots(bin_scale_factor=6, plt_red_factor = 1)
postprs1.show_plots()


# postprs3 = PostProcS3("/home/andreap/hhcm_ws/src/iit-repair-ros-pkg/repair_codesign/test_results/test_results_11-10-2022-14_19_55") 